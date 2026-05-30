def schedule_pipeline(tasks, resource_budget):
    """
    Schedule ETL tasks respecting dependencies and resource limits.
    """
    # Write code here
    remaining_tasks = [task for task in tasks if task["resources"] <= resource_budget]
    completed_tasks = set()
    running_tasks = []
    current_time = 0
    schedule = []
    while remaining_tasks or running_tasks:
        still_running = []
        for task in running_tasks:
            if task["end_time"] <= current_time:
                completed_tasks.add(task["name"])
            else:
                still_running.append(task)
        running_tasks = still_running
        used_resource = sum(task["resources"] for task in still_running)
        ready_tasks = []
        for task in remaining_tasks:
            if all(dep in completed_tasks for dep in task["depends_on"]):
                ready_tasks.append(task)
        ready_tasks.sort(key=lambda x: x["name"])
        task_to_remove = []
        for task in ready_tasks:
            if used_resource + task["resources"] <= resource_budget:
                start_time = current_time
                end_time = start_time + task["duration"]

                schedule.append((task["name"], start_time))
                running_tasks.append({
                    "name": task["name"],
                    "end_time": end_time,
                    "resources": task["resources"]
                })
                used_resource += task["resources"]
                task_to_remove.append(task)
        for task in task_to_remove:
            remaining_tasks.remove(task)

        if running_tasks:
            next_time = min(t["end_time"] for t in running_tasks)
            current_time = next_time
        else:
            if remaining_tasks:
                current_time += 1
    schedule.sort(key=lambda x: (x[1], x[0]))
    return schedule