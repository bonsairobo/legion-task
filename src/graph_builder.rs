use crate::components::*;

use legion::prelude::*;

pub trait TaskFactory {
    fn create_task(&self, cmd: &mut CommandBuffer) -> Entity;
}

impl<'a, T: 'static + Clone + TaskComponent<'a>> TaskFactory for T {
    fn create_task(&self, cmd: &mut CommandBuffer) -> Entity {
        make_task(cmd, self.clone())
    }
}

// TODO: a simple way to accumulate TaskGraph in a loop without using Option<TaskGraph>

/// Implementation detail of `TaskGraph`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Cons<T> {
    Fork(Box<Cons<T>>, Box<Cons<T>>),
    Seq(Box<Cons<T>>, Box<Cons<T>>),
    Task(T),
}

/// A node of the binary tree grammar that describes a task graph. `Cons::Seq` lists represent
/// sequential execution of tasks. `Cons::Fork` lists represent concurrent execution of tasks. The
/// leaves of the tree are `Cons::Task`s.
pub type TaskGraph = Cons<Box<dyn TaskFactory + Send + Sync>>;

impl Cons<Box<dyn TaskFactory + Send + Sync>> {
    fn _assemble(self, fork: Option<Entity>, cmd: &mut CommandBuffer) -> (Entity, Entity) {
        match self {
            Cons::Seq(head, tail) => {
                let (head_first_entity, head_last_entity) = head._assemble(None, cmd);
                let (tail_first_entity, tail_last_entity) = tail._assemble(None, cmd);
                join(cmd, tail_first_entity, head_last_entity);

                (head_first_entity, tail_last_entity)
            }
            Cons::Fork(head, tail) => {
                let fork_entity = if let Some(e) = fork {
                    e
                } else {
                    make_fork(cmd)
                };

                let (_, head_last_entity) = head._assemble(Some(fork_entity), cmd);
                let (_, tail_last_entity) = tail._assemble(Some(fork_entity), cmd);

                // Any decendents reachable only via Cons::Fork are considered prongs. If a
                // descendent is a Cons::Seq, then the prong only connects at the "last" entity of
                // the sequence.
                if head_last_entity != fork_entity {
                    add_prong(cmd, fork_entity, head_last_entity);
                }
                if tail_last_entity != fork_entity {
                    add_prong(cmd, fork_entity, tail_last_entity);
                }

                (fork_entity, fork_entity)
            }
            Cons::Task(task) => {
                let task_entity = task.create_task(cmd);

                (task_entity, task_entity)
            }
        }
    }

    /// Mark the root of the `TaskGraph` as final, effectively unblocking the first tasks in this
    /// graph to be run.
    pub fn assemble(self, on_completion: OnCompletion, cmd: &mut CommandBuffer) -> Entity {
        let (_first_entity, last_entity) = self._assemble(None, cmd);
        finalize(cmd, last_entity, on_completion);

        last_entity
    }
}

// TODO: Get rid of the "@" that precedes every task expression. I am bad at macros, please help!

/// Make a single-node `TaskGraph`.
#[macro_export]
macro_rules! task {
    (@$task:expr) => {
        Cons::Task(Box::new($task))
    };
}

// TODO: deduplicate these definitions that are mostly the same

/// Returns a `TaskGraph` that executes the argument list of `TaskGraphs` concurrently.
#[macro_export]
macro_rules! fork {
    (@$head:expr, $($tail:tt)*) => (
        Cons::Fork(Box::new(fork!(@$head)), Box::new(fork!($($tail)*)))
    );
    ($head:expr, $($tail:tt)*) => (
        Cons::Fork(Box::new(fork!($head)), Box::new(fork!($($tail)*)))
    );
    (@$task:expr) => (
        Cons::Task(Box::new($task))
    );
    ($head:expr) => ( $head );
}

/// Returns a `TaskGraph` that executes the argument list of `TaskGraphs` sequentially.
#[macro_export]
macro_rules! seq {
    (@$head:expr, $($tail:tt)*) => (
        Cons::Seq(Box::new(seq!(@$head)), Box::new(seq!($($tail)*)))
    );
    ($head:expr, $($tail:tt)*) => (
        Cons::Seq(Box::new(seq!($head)), Box::new(seq!($($tail)*)))
    );
    (@$task:expr) => (
        Cons::Task(Box::new($task))
    );
    ($head:expr) => ( $head );
}

mod tests {
    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct Foo(u32);

    #[test]
    fn task_graph_macro_fork_two() {
        let x = fork!(@Foo(1u32), @Foo(2u32));
        assert_eq!(
            x,
            Cons::Fork(
                Box::new(Cons::Task(Box::new(Foo(1u32)))),
                Box::new(Cons::Task(Box::new(Foo(2u32)))),
            )
        );
    }

    #[test]
    fn task_graph_macro_fork_three() {
        let x = fork!(@Foo(1u32), @Foo(2u32), @Foo(3u32));
        assert_eq!(
            x,
            Cons::Fork(
                Box::new(Cons::Task(Box::new(Foo(1u32)))),
                Box::new(Cons::Fork(
                    Box::new(Cons::Task(Box::new(Foo(2u32)))),
                    Box::new(Cons::Task(Box::new(Foo(3u32)))),
                )),
            )
        );
    }

    #[test]
    fn task_graph_macro_nested_fork() {
        let x: Cons<Box<Foo>> = fork!(@Foo(1u32), fork!(@Foo(2u32), @Foo(3u32)));
        assert_eq!(
            x,
            Cons::Fork(
                Box::new(Cons::Task(Box::new(Foo(1u32)))),
                Box::new(Cons::Fork(
                    Box::new(Cons::Task(Box::new(Foo(2u32)))),
                    Box::new(Cons::Task(Box::new(Foo(3u32)))),
                )),
            )
        );
    }

    #[test]
    fn task_graph_macro_many_nested() {
        let x = seq!(
            @Foo(1u32),
            fork!(
                seq!(@Foo(2u32), @Foo(3u32), @Foo(4u32)),
                @Foo(5u32),
                @Foo(6u32)
            ),
            @Foo(7u32)
        );
        let y: Cons<Box<Foo>> = Cons::Seq(
            Box::new(Cons::Task(Box::new(Foo(1u32)))),
            Box::new(Cons::Seq(
                Box::new(Cons::Fork(
                    Box::new(Cons::Seq(
                        Box::new(Cons::Task(Box::new(Foo(2u32)))),
                        Box::new(Cons::Seq(
                            Box::new(Cons::Task(Box::new(Foo(3u32)))),
                            Box::new(Cons::Task(Box::new(Foo(4u32)))),
                        )),
                    )),
                    Box::new(Cons::Fork(
                        Box::new(Cons::Task(Box::new(Foo(5u32)))),
                        Box::new(Cons::Task(Box::new(Foo(6u32)))),
                    )),
                )),
                Box::new(Cons::Task(Box::new(Foo(7u32)))),
            )),
        );
        assert_eq!(x, y);
    }
}
