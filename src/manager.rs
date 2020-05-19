use crate::components::{FinalTag, MultiEdge, OnCompletion, SingleEdge, TaskProgress};

use legion::{
    prelude::*,
    systems::{SubWorld, SystemId},
};

/// Returns true iff the task was seen as complete on the last run of the `TaskManagerSystem`.
///
/// WARNING: assumes that this entity was at one point a task, and it can't tell otherwise.
pub fn task_is_complete(world: &SubWorld, entity: Entity) -> bool {
    world.get_component::<TaskProgress>(entity).is_none()
}

/// Returns true iff all of `entity`'s children are complete.
pub fn fork_is_complete(world: &SubWorld, entity: Entity, multi_children: &[Entity]) -> bool {
    if let Some(edge) = world.get_component::<SingleEdge>(entity) {
        if !entity_is_complete(world, edge.child) {
            return false;
        }
    }
    for child in multi_children.iter() {
        if !entity_is_complete(world, *child) {
            return false;
        }
    }

    true
}

/// Tells you whether a fork or a task entity is complete.
///
/// WARNING: assumes that this entity was at one point a task or a fork, and it can't tell
/// otherwise.
pub fn entity_is_complete(world: &SubWorld, entity: Entity) -> bool {
    // Only fork entities can have `MultiEdge`s, and they always do.
    if let Some(edge) = world.get_component::<MultiEdge>(entity) {
        fork_is_complete(world, entity, &edge.children)
    } else {
        task_is_complete(world, entity)
    }
}

/// Deletes only the descendent entities of `entity`, but leaves `entity` alive.
pub fn delete_descendents(cmd: &CommandBuffer, world: &SubWorld, entity: Entity) {
    if let Some(edge) = world.get_component::<MultiEdge>(entity) {
        for child in edge.children.iter() {
            delete_entity_and_descendents(cmd, world, *child);
        }
    }
    if let Some(edge) = world.get_component::<SingleEdge>(entity) {
        delete_entity_and_descendents(cmd, world, edge.child);
    }
}

/// Deletes `entity` and all of its descendents.
pub fn delete_entity_and_descendents(cmd: &CommandBuffer, world: &SubWorld, entity: Entity) {
    // Support async deletion. If a child is deleted, we assume all of its descendants were also
    // deleted.
    if !world.is_alive(entity) {
        return;
    }

    delete_descendents(cmd, world, entity);
    log::debug!("Deleting {:?}", entity);
    cmd.delete(entity);
}

/// Returns `true` iff `entity` is complete.
fn maintain_task_and_descendents(
    cmd: &CommandBuffer,
    world: &mut SubWorld,
    entity: Entity,
) -> bool {
    let (is_unblocked, is_complete) =
        if let Some(progress) = world.get_component::<TaskProgress>(entity) {
            (progress.is_unblocked, progress.is_complete())
        } else {
            // Missing progress means the task is complete and progress was already removed.
            return true;
        };

    if is_complete {
        log::debug!(
            "Noticed task {:?} is complete, removing TaskProgress",
            entity
        );
        // Task will no longer be considered by the `TaskRunnerSystem`.
        // PERF: avoid this?
        cmd.remove_component::<TaskProgress>(entity);
        return true;
    }

    // If `is_unblocked`, the children don't need maintenance, because we already verified they
    // are all complete.
    if is_unblocked {
        return false;
    }

    // Unblock the task if its child is complete.
    let mut child_complete = true;
    if let Some(edge) = world
        .get_component::<SingleEdge>(entity)
        .map(|e| (*e).clone())
    {
        child_complete = maintain_entity_and_descendents(cmd, world, edge.child);
    }
    if child_complete {
        log::debug!("Unblocking task {:?}", entity);
        let mut progress = world
            .get_component_mut::<TaskProgress>(entity)
            .expect("Blocked task must have progress");
        progress.unblock();
    }

    false
}

/// Returns `true` iff `entity` is complete.
fn maintain_fork_and_descendents(
    cmd: &CommandBuffer,
    world: &mut SubWorld,
    entity: Entity,
    multi_edge_children: &[Entity],
) -> bool {
    // We make sure that the SingleEdge child completes before any of the MultiEdge descendents
    // can start.
    let mut single_child_complete = true;
    if let Some(edge) = world
        .get_component::<SingleEdge>(entity)
        .map(|e| (*e).clone())
    {
        single_child_complete = maintain_entity_and_descendents(cmd, world, edge.child);
    }
    let mut multi_children_complete = true;
    if single_child_complete {
        for child in multi_edge_children.iter() {
            multi_children_complete &= maintain_entity_and_descendents(cmd, world, *child);
        }
    }

    single_child_complete && multi_children_complete
}

/// Returns `true` iff `entity` is complete.
fn maintain_entity_and_descendents(
    cmd: &CommandBuffer,
    world: &mut SubWorld,
    entity: Entity,
) -> bool {
    // Only fork entities can have `MultiEdge`s, and they always do.
    if let Some(edge) = world
        .get_component::<MultiEdge>(entity)
        .map(|e| (*e).clone())
    {
        maintain_fork_and_descendents(cmd, world, entity, &edge.children)
    } else {
        maintain_task_and_descendents(cmd, world, entity)
    }
}

/// Creates a system that traverses all descendents of all finalized entities and unblocks them if
/// possible.
///
/// Also does some garbage collection:
///   - removes `TaskProgress` components from completed tasks
///   - deletes task graphs with `OnCompletion::Delete`
///   - removes `FinalTag` components from completed entities
pub fn build_task_manager_system<I: Into<SystemId>>(id: I) -> Box<dyn Schedulable> {
    SystemBuilder::new(id)
        .read_component::<MultiEdge>()
        .write_component::<MultiEdge>()
        .read_component::<SingleEdge>()
        .write_component::<SingleEdge>()
        .read_component::<TaskProgress>()
        .write_component::<TaskProgress>()
        .with_query(<Read<FinalTag>>::query())
        .build(|cmd, mut world, _, final_tasks_query| {
            let final_entities: Vec<(Entity, FinalTag)> = final_tasks_query
                .iter_entities(&world)
                .map(|(e, f)| (e, *f))
                .collect();

            for (entity, FinalTag { on_completion }) in final_entities.into_iter() {
                let final_complete = maintain_entity_and_descendents(cmd, &mut world, entity);
                if final_complete {
                    match on_completion {
                        OnCompletion::Delete => {
                            delete_entity_and_descendents(cmd, &world, entity);
                        }
                        OnCompletion::DeleteDescendents => {
                            delete_descendents(cmd, &world, entity);
                        }
                        OnCompletion::None => {
                            log::debug!("Removing FinalTag from {:?}", entity);
                            // PERF: avoid this?
                            cmd.remove_component::<FinalTag>(entity);
                        }
                    }
                }
            }
        })
}
