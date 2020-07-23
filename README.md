# Fork-join multitasking for Legion ECS

Instead of hand-rolling state machines to sequence the effects of various ECS systems, spawn
tasks as entities and declare explicit temporal dependencies between them.

## Code Example: making task graphs and dispatching task runners
```rust
use legion::prelude::*;
use legion_task::*;

#[derive(Clone)]
struct SaySomething(&'static str);
impl<'a> TaskComponent<'a> for SaySomething {
    type Data = ();
    fn run(&mut self, data: &mut Self::Data) -> bool {
        println!("{}", self.0);
        true
    }
}

#[derive(Clone, Debug)]
struct PushValue {
    value: usize,
}

impl<'a> TaskComponent<'a> for PushValue {
    type Data = Vec<usize>;
    fn run(&mut self, data: &mut Self::Data) -> bool {
        data.push(self.value);
        true
    }
}

fn make_static_task_graph(cmd: &mut CommandBuffer) {
    // Any component that implements TaskComponent can be spawned.
    let task_graph: TaskGraph = seq!(
        @SaySomething("hello"),
        fork!(
            @PushValue { value: 1 },
            @PushValue { value: 2 },
            @PushValue { value: 3 }
        ),
        @SaySomething("goodbye")
    );
    task_graph.assemble(OnCompletion::Delete, cmd);
}

fn make_dynamic_task_graph(cmd: &mut CommandBuffer) {
    let first: TaskGraph = task!(@SaySomething("hello"));
    let mut middle: TaskGraph = empty_graph!();
    for i in 0..10 {
        middle = fork!(middle, @PushValue { value: i });
    }
    let last: TaskGraph = task!(@SaySomething("goodbye"));
    let task_graph: TaskGraph = seq!(first, middle, last);
    task_graph.assemble(OnCompletion::Delete, cmd);
}

fn build_say_something_task_runner_system() -> Box<dyn Schedulable> {
    SystemBuilder::new("say_something_task_runner")
        .with_query(task_runner_query::<SaySomething>())
        .build(|_, mut world, _, task_query| {
            run_tasks(&mut world, &mut (), task_query)
        })
}

fn build_push_value_task_runner_system() -> Box<dyn Schedulable> {
    SystemBuilder::new("push_value_task_runner")
        .write_resource::<Vec<usize>>()
        .with_query(task_runner_query::<PushValue>())
        .build(|_, mut world, value, task_query| {
            run_tasks(&mut world, &mut **value, task_query)
        })
}

fn make_schedule() -> Schedule {
    Schedule::builder()
        .add_system(build_say_something_task_runner_system())
        .add_system(build_push_value_task_runner_system())
        .add_system(build_task_manager_system("task_manager"))
        .build()
}
```
