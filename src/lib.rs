//! # Fork-join multitasking for Legion ECS
//!
//! Instead of hand-rolling state machines to sequence the effects of various ECS systems, spawn
//! tasks as entities and declare explicit temporal dependencies between them.
//!
//! ## Code Examples
//!
//! ```compile_fail
//! fn make_static_task_graph(cmd: &mut CommandBuffer) {
//!     // Any component that implements TaskComponent can be spawned.
//!     let task_graph = seq!(
//!         @TaskFoo("hello"),
//!         fork!(@TaskBar { value: 2 }, @TaskBar { value: 3 }),
//!         @TaskZing("goodbye")
//!     );
//!     task_graph.assemble(cmd, OnCompletion::Delete);
//! }
//!
//! fn make_dynamic_task_graph(cmd: &mut CommandBuffer) {
//!     let first = task!(@TaskFoo("hello"));
//!     let mut middle = empty_graph!();
//!     for i in 0..10 {
//!         middle = fork!(middle, @TaskBar { value: i });
//!     }
//!     let last = task!(@TaskZin("goodbye"));
//!     let task_graph = seq!(first, middle, last);
//!     task_graph.assemble(cmd, OnCompletion::Delete);
//! }
//! ```
//!
//! ## Data Model
//!
//! Here we expound on the technical details of this module's implementation. For basic usage, see
//! the tests.
//!
//! In this model, every task is some entity. The entity is allowed to have exactly one component
//! that implements `TaskComponent` (it may have other components that don't implement
//! `TaskComponent`). The task will be run to completion by running a system that calls `run_tasks`
//! with the proper `TaskComponent::Data` and `task_query`.
//!
//! Every task entity is also a node in a (hopefully acyclic) directed graph. An edge `t2 --> t1`
//! means that `t2` cannot start until `t1` has completed.
//!
//! In order for tasks to become unblocked, the system created with `build_task_manager_system` must
//! run, whence it will traverse the graph, starting at the "final entities", and check for entities
//! that have completed, potentially unblocking their parents. In order for a task to be run, it
//! must be the descendent of a final entity. Entity component tuples become final by calling
//! `finalize` (which adds a `FinalTag` component).
//!
//! Edges can either come from `SingleEdge` or `MultiEdge` components, but you should not use these
//! types directly. You might wonder why we need both types of edges. It's a fair question, because
//! adding the `SingleEdge` concept does not actually make the model capable of representing any
//! semantically new graphs. The reason is efficiency.
//!
//! If you want to implement a fork join like this (note: time is going left to right but the
//! directed edges are going right to left):
//!
//!```
//! r#"       ----- t1.1 <---   ----- t2.1 <---
//!          /               \ /               \
//!      t0 <------ t1.2 <----<------ t2.2 <---- t3
//!          \               / \               /
//!           ----- t1.3 <---   ----- t2.3 <---      "#;
//!```
//!
//! You would actually do this by calling `make_fork` to create two "fork" entities `F1` and `F2`
//! that don't have `TaskComponent`s, but they can have both a `SingleEdge` and a `MultiEdge`. Note
//! that the children on the `MultiEdge` are called "prongs" of the fork.
//!
//!```
//! r#"      single          single          single
//!     t0 <-------- F1 <-------------- F2 <-------- t3
//!                   |                  |
//!          t1.1 <---|          t2.1 <--|
//!          t1.2 <---| multi    t2.2 <--| multi
//!          t1.3 <---|          t2.3 <--|            "#;
//!```
//!
//! The semantics would be such that this graph is equivalent to the one above. Before any of the
//! tasks connected to `F2` by the `MultiEdge` could run, the tasks connected by the `SingleEdge`
//! (`{ t0, t1.1, t1.2, t1.3 }`) would have to be complete. `t3` could only run once all of the
//! descendents of `F2` had completed.
//!
//! The advantages of this scheme are:
//!   - a traversal of the graph starting from `t3` does not visit the same node twice
//!   - it is a bit easier to create fork-join graphs with larger numbers of concurrent tasks
//!   - there are fewer edges for the most common use cases
//!
//! Here's another example with "nested forks" to test your understanding:
//!
//! ```
//! r#"   With fork entities:
//!
//!           t0 <-------------- FA <----- t2
//!                              |
//!                       tx <---|
//!               t1 <--- FB <---|
//!                        |
//!               ty <-----|
//!               tz <-----|
//!
//!       As time orderings:
//!
//!           t0   < { t1, tx, ty, tz } < t2
//!           t1   < { ty, tz }
//!
//!       Induced graph:
//!
//!           t0 <------- tx <------- t2
//!            ^                      |
//!            |      /------ ty <----|
//!            |     v                |
//!            ----- t1 <---- tz <-----          "#;
//! ```
//!
//! ## Macro Usage
//!
//! Every user of this module should create task graphs via the `empty_graph!`, `seq!`, `fork!`, and
//! `task!` macros, which make it easy to construct task graphs correctly. Once a graph is ready,
//! call `assemble` on it to mark the task entities for execution (by finalizing the root of the
//! graph).
//!
//! These systems must be scheduled for tasks to make progress:
//!   - a system created with `build_task_manager_system`
//!   - a system that calls `run_tasks` on each `TaskComponent` used
//!
//! ## Advanced Usage
//!
//! If you find the `TaskGraph` macros limiting, you can use the `make_task`, `join`, `make_fork`,
//! and `add_prong` functions; these are the building blocks for creating all task graphs, including
//! buggy ones. These functions are totally dynamic in that they deal directly with entities of
//! various archetypes, assuming that the programmer passed in the correct archetypes for the given
//! function.
//!
//! Potential bugs that won't be detected for you:
//!   - leaked orphan entities
//!   - graph cycles
//!   - finalizing an entity that has children
//!   - users manually tampering with the `TaskProgress`, `SingleEdge`, `MultiEdge`, or `FinalTag`
//!     components; these should only be used inside this module
//!

#[macro_use]
mod graph_builder;

mod components;
mod manager;
mod runner;

pub use components::{
    add_prong, finalize, join, make_fork, make_task, with_task_components, FinalTag, OnCompletion,
    TaskComponent, TaskProgress,
};
pub use graph_builder::{Cons, TaskFactory, TaskGraph};
pub use manager::{build_task_manager_system, entity_is_complete};
pub use runner::{run_tasks, task_runner_query, TaskEntityFilter, TaskQuery};

#[cfg(test)]
mod tests {
    use super::*;

    use legion::prelude::*;

    #[derive(Clone, Debug, Default, Eq, PartialEq)]
    struct Noop {
        was_run: bool,
    }

    impl<'a> TaskComponent<'a> for Noop {
        type Data = ();

        fn run(&mut self, _data: &mut Self::Data) -> bool {
            self.was_run = true;

            true
        }
    }

    fn build_noop_task_runner_system() -> Box<dyn Schedulable> {
        SystemBuilder::new("noop_task_runner")
            .with_query(task_runner_query::<Noop>())
            .build(|_, mut world, _, task_query| run_tasks(&mut world, &mut (), task_query))
    }

    #[derive(Clone, Debug)]
    struct PushValue {
        value: usize,
    }

    impl<'a> TaskComponent<'a> for PushValue {
        type Data = Vec<usize>;

        fn run(&mut self, data: &mut Self::Data) -> bool {
            log::debug!("Task pushing value {}", self.value);
            data.push(self.value);

            true
        }
    }

    fn build_push_value_task_runner_system() -> Box<dyn Schedulable> {
        SystemBuilder::new("example_task_runner")
            .write_resource::<Vec<usize>>()
            .with_query(task_runner_query::<PushValue>())
            .build(|_, mut world, value, task_query| {
                run_tasks(&mut world, &mut **value, task_query)
            })
    }

    fn set_up<'a, 'b>() -> (World, Resources, Schedule) {
        let mut resources = Resources::default();
        resources.insert::<Vec<usize>>(Vec::new());

        let world = World::new();

        let schedule = Schedule::builder()
            .add_system(build_noop_task_runner_system())
            .add_system(build_push_value_task_runner_system())
            // For sake of reproducible tests, assume the manager system is the last to run.
            .add_system(build_task_manager_system("task_manager"))
            .build();

        (world, resources, schedule)
    }

    fn assemble_task_graph(
        make_task_graph: fn() -> TaskGraph,
        on_completion: OnCompletion,
        world: &mut World,
        resources: &mut Resources,
    ) -> Entity {
        resources.insert::<Option<Entity>>(None);
        let assemble_system = SystemBuilder::new("assembler")
            .write_resource::<Option<Entity>>()
            .build(move |mut cmd, _subworld, final_task, _| {
                **final_task = Some(make_task_graph().assemble(on_completion, &mut cmd));
            });
        let mut assemble_schedule = Schedule::builder()
            .add_system(assemble_system)
            .flush()
            .build();
        assemble_schedule.execute(world, resources);

        resources.get::<Option<Entity>>().unwrap().unwrap()
    }

    fn assert_task_is_complete(
        task: Entity,
        is_alive: bool,
        world: &mut World,
        resources: &mut Resources,
    ) {
        let assert_system =
            with_task_components(SystemBuilder::new("asserter")).build(move |_, subworld, _, _| {
                assert!(entity_is_complete(&subworld, task));
                assert_eq!(subworld.is_alive(task), is_alive);
            });
        let mut assert_schedule = Schedule::builder().add_system(assert_system).build();
        assert_schedule.execute(world, resources);
    }

    #[test]
    fn run_single_task() {
        let (mut world, mut resources, mut schedule) = set_up();

        fn make_task_graph() -> TaskGraph {
            task!(@Noop::default())
        }
        let root = assemble_task_graph(
            make_task_graph,
            OnCompletion::None,
            &mut world,
            &mut resources,
        );

        schedule.execute(&mut world, &mut resources);
        schedule.execute(&mut world, &mut resources);

        assert_eq!(
            *world.get_component::<Noop>(root).unwrap(),
            Noop { was_run: true }
        );
        assert_task_is_complete(root, true, &mut world, &mut resources);
    }

    #[test]
    fn single_task_deleted_on_completion() {
        let (mut world, mut resources, mut schedule) = set_up();

        fn make_task_graph() -> TaskGraph {
            task!(@Noop::default())
        }
        let root = assemble_task_graph(
            make_task_graph,
            OnCompletion::Delete,
            &mut world,
            &mut resources,
        );

        schedule.execute(&mut world, &mut resources);
        schedule.execute(&mut world, &mut resources);

        assert_task_is_complete(root, false, &mut world, &mut resources);
    }

    #[test]
    fn joined_tasks_run_in_order_and_deleted_on_completion() {
        let (mut world, mut resources, mut schedule) = set_up();

        fn make_task_graph() -> TaskGraph {
            seq!(
                @PushValue { value: 1 },
                @PushValue { value: 2 },
                @PushValue { value: 3 }
            )
        }
        let root = assemble_task_graph(
            make_task_graph,
            OnCompletion::Delete,
            &mut world,
            &mut resources,
        );

        schedule.execute(&mut world, &mut resources);
        schedule.execute(&mut world, &mut resources);
        schedule.execute(&mut world, &mut resources);
        schedule.execute(&mut world, &mut resources);

        assert_eq!(*resources.get::<Vec<usize>>().unwrap(), vec![1, 2, 3]);
        assert_task_is_complete(root, false, &mut world, &mut resources);
    }

    #[test]
    fn all_prongs_of_fork_run_before_join_and_deleted_on_completion() {
        let (mut world, mut resources, mut schedule) = set_up();

        //         ---> t1.1 ---
        //       /               \
        //     t2 ----> t1.2 -----> t0

        fn make_task_graph() -> TaskGraph {
            seq!(
                @PushValue { value: 1 },
                fork!(@PushValue { value: 2 }, @PushValue { value: 3 }),
                @PushValue { value: 4 }
            )
        }
        let root = assemble_task_graph(
            make_task_graph,
            OnCompletion::Delete,
            &mut world,
            &mut resources,
        );

        schedule.execute(&mut world, &mut resources);
        schedule.execute(&mut world, &mut resources);
        schedule.execute(&mut world, &mut resources);
        schedule.execute(&mut world, &mut resources);

        let pushed_values: Vec<usize> = (*resources.get::<Vec<usize>>().unwrap()).clone();
        assert!(pushed_values == vec![1, 2, 3, 4] || pushed_values == vec![1, 3, 2, 4]);

        assert_task_is_complete(root, false, &mut world, &mut resources);
    }

    #[test]
    fn join_fork_with_nested_fork() {
        let (mut world, mut resources, mut schedule) = set_up();

        fn make_task_graph() -> TaskGraph {
            seq!(
                @PushValue { value: 1 },
                fork!(
                    @PushValue { value: 2 },
                    fork!(@PushValue { value: 3 }, @PushValue { value: 4 })
                ),
                @PushValue { value: 5 }
            )
        }
        let root = assemble_task_graph(
            make_task_graph,
            OnCompletion::Delete,
            &mut world,
            &mut resources,
        );

        schedule.execute(&mut world, &mut resources);
        schedule.execute(&mut world, &mut resources);
        schedule.execute(&mut world, &mut resources);
        schedule.execute(&mut world, &mut resources);

        let pushed_values: Vec<usize> = (*resources.get::<Vec<usize>>().unwrap()).clone();
        assert!(
            pushed_values == vec![1, 2, 3, 4, 5]
                || pushed_values == vec![1, 2, 4, 3, 5]
                || pushed_values == vec![1, 3, 2, 4, 5]
                || pushed_values == vec![1, 3, 4, 2, 5]
                || pushed_values == vec![1, 4, 2, 3, 5]
                || pushed_values == vec![1, 4, 3, 2, 5]
        );

        assert_task_is_complete(root, false, &mut world, &mut resources);
    }
}
