use crate::components::{TaskComponent, TaskProgress};

use legion::{
    filter::{And, ComponentFilter, EntityFilterTuple, Passthrough},
    prelude::*,
    systems::{SubWorld, SystemQuery},
};

/// The type of `SystemQuery` created by `task_runner_query` and used by `run_tasks`.
pub type TaskSystemQuery<T> = SystemQuery<(Read<TaskProgress>, Write<T>), TaskEntityFilter<T>>;

/// The type of `Query` created by `task_runner_query` and used by `run_tasks`.
pub type TaskQuery<T> = Query<(Read<TaskProgress>, Write<T>), TaskEntityFilter<T>>;

/// The `EntityFilterTuple` for `task_runner_query`.
pub type TaskEntityFilter<T> = EntityFilterTuple<
    And<(ComponentFilter<TaskProgress>, ComponentFilter<T>)>,
    And<(Passthrough, Passthrough)>,
    And<(Passthrough, Passthrough)>,
>;

/// Run the tasks that match `task_query`. Should be run in a `System` created with
/// `task_runner_query`.
pub fn run_tasks<'a, T: 'static + TaskComponent<'a>>(
    world: &mut SubWorld,
    task_component_data: &mut T::Data,
    task_query: &mut TaskSystemQuery<T>,
) {
    for (task_progress, mut task) in task_query.iter_mut(world) {
        if !task_progress.is_unblocked || task_progress.is_complete() {
            continue;
        }
        let is_complete = task.run(task_component_data);
        if is_complete {
            task_progress.complete();
        }
    }
}

/// The legion system query required to run all tasks with `T: TaskComponent`.
pub fn task_runner_query<'a, T: 'static + TaskComponent<'a>>(
) -> TaskQuery<T> {
    <(Read<TaskProgress>, Write<T>)>::query()
}
