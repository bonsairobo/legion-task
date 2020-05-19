use legion::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// An ephemeral component that needs access to `Data` to run some task. Will be run by `run_tasks`
/// in a system with access to `task_runner_query` and `Data`.
pub trait TaskComponent<'a>: Send + Sync {
    type Data;

    /// Returns `true` iff the task is complete.
    fn run(&mut self, data: &mut Self::Data) -> bool;
}

#[doc(hidden)]
#[derive(Default)]
pub struct TaskProgress {
    pub(crate) is_complete: AtomicBool,
    pub(crate) is_unblocked: bool,
}

impl TaskProgress {
    pub(crate) fn is_complete(&self) -> bool {
        self.is_complete.load(Ordering::Relaxed)
    }

    pub(crate) fn complete(&self) {
        self.is_complete.store(true, Ordering::Relaxed);
    }

    pub(crate) fn unblock(&mut self) {
        self.is_unblocked = true;
    }
}

#[doc(hidden)]
#[derive(Clone)]
pub struct SingleEdge {
    pub(crate) child: Entity,
}

#[doc(hidden)]
#[derive(Clone, Default)]
pub struct MultiEdge {
    pub(crate) children: Vec<Entity>,
}

impl MultiEdge {
    fn add_child(&mut self, entity: Entity) {
        self.children.push(entity);
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Default)]
pub struct FinalTag {
    pub(crate) on_completion: OnCompletion,
}

/// What to do to a final task and its descendents when they complete.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum OnCompletion {
    None,
    Delete,
    DeleteDescendents,
}

impl Default for OnCompletion {
    fn default() -> Self {
        OnCompletion::None
    }
}

/// Gives read-only access to the task meta-components in order to query the state of task entities.
pub fn with_task_components(builder: SystemBuilder) -> SystemBuilder {
    builder
        .read_component::<TaskProgress>()
        .read_component::<SingleEdge>()
        .read_component::<MultiEdge>()
}

/// Create a new task entity.
pub fn make_task<'a, T: 'static + TaskComponent<'a>>(
    cmd: &mut CommandBuffer,
    task_component: T,
) -> Entity {
    let entity = cmd
        .start_entity()
        .with_component(TaskProgress::default())
        .with_component(task_component)
        .build();
    log::debug!("Created task {:?}", entity);

    entity
}

/// Mark `entity` as "final," i.e. a task with no parent.
pub fn finalize(cmd: &CommandBuffer, entity: Entity, on_completion: OnCompletion) {
    cmd.add_component(entity, FinalTag { on_completion });
    log::debug!("Finalized task {:?}", entity);
}

/// Create a new fork entity with no children.
pub fn make_fork(cmd: &mut CommandBuffer) -> Entity {
    let entity = cmd
        .start_entity()
        .with_component(MultiEdge::default())
        // BUG: builder seems to require at least 2 components
        .with_component(())
        .build();
    log::debug!("Created fork {:?}", entity);

    entity
}

/// Add `prong` as a child on the `MultiEdge` of `fork_entity`.
pub fn add_prong(cmd: &CommandBuffer, fork_entity: Entity, prong: Entity) {
    cmd.exec_mut(move |world| {
        let mut multi_edge = world
            .get_component_mut::<MultiEdge>(fork_entity)
            .unwrap_or_else(|| {
                panic!(
                    "Tried to add prong {} to non-fork entity {}",
                    prong, fork_entity
                )
            });
        multi_edge.add_child(prong);
    });
    log::debug!(
        "Submitted command to add prong {} to fork {}",
        prong,
        fork_entity
    );
}

/// Creates a `SingleEdge` from `parent` to `child`. Creates a fork-join if `parent` is a fork.
pub fn join(cmd: &CommandBuffer, parent: Entity, child: Entity) {
    cmd.exec_mut(move |world| {
        if let Some(edge) = world
            .get_component::<SingleEdge>(parent)
            .map(|e| (*e).clone())
        {
            panic!(
                "Attempted to make task {} child of {}, but task {} already has child {}",
                child, parent, parent, edge.child
            );
        } else {
            // PERF: avoid this?
            world.add_component(parent, SingleEdge { child }).unwrap();
        }
    });
    log::debug!("Submitted command to make {} parent of {}", parent, child);
}
