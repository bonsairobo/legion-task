# Fork-join multitasking for Legion ECS

Instead of hand-rolling state machines to sequence the effects of various ECS systems, spawn
tasks as entities and declare explicit temporal dependencies between them.

## Code Examples

### Making task graphs

```rust
fn make_static_task_graph(cmd: &mut CommandBuffer) {
    // Any component that implements TaskComponent can be spawned.
    let task_graph = seq!(
        @TaskFoo("hello"),
        fork!(
            @TaskBar { value: 1 },
            @TaskBar { value: 2 },
            @TaskBar { value: 3 }
        ),
        @TaskZing("goodbye")
    );
    task_graph.assemble(cmd, OnCompletion::Delete);
}

fn make_dynamic_task_graph(cmd: &mut CommandBuffer) {
    let first = task!(@TaskFoo("hello"));
    let mut middle = empty_graph!();
    for i in 0..10 {
        middle = fork!(middle, @TaskBar { value: i });
    }
    let last = task!(@TaskZing("goodbye"));
    let task_graph = seq!(first, middle, last);
    task_graph.assemble(cmd, OnCompletion::Delete);
}
```

### Building a schedule with a task runner system

```rust
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

fn make_task_runner_system() -> Box<dyn Schedulable> {
    SystemBuilder::new("example_task_runner")
        .write_resource::<Vec<usize>>()
        .with_query(task_runner_query::<PushValue>())
        .build(|_, mut world, value, task_query| {
            run_tasks(&mut world, &mut **value, task_query)
        })
}

fn make_schedule() -> Schedule {
    Schedule::builder()
        .add_system(build_push_value_task_runner_system())
        .add_system(build_task_manager_system("task_manager"))
        .build()
}
```
