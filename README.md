# Fork-join multitasking for Legion ECS

Instead of hand-rolling state machines to sequence the effects of various ECS systems, spawn
tasks as entities and declare explicit temporal dependencies between them.

## Code Examples

```
fn make_static_task_graph(cmd: &mut CommandBuffer) {
    // Any component that implements TaskComponent can be spawned.
    let task_graph = seq!(
        @TaskFoo("hello"),
        fork!(@TaskBar { value: 2 }, @TaskBar { value: 3 }),
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
    let last = task!(@TaskZin("goodbye"));
    let task_graph = seq!(first, middle, last);
    task_graph.assemble(cmd, OnCompletion::Delete);
}
```
