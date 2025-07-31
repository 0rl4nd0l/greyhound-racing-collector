from locust import HttpUser, TaskSet, between, task


class UserBehavior(TaskSet):
    @task(1)
    def predictions_upcoming(self):
        self.client.post(
            "/api/predictions/upcoming", json={"race_ids": ["race_001", "race_002"]}
        )

    @task(1)
    def upcoming_races_stream(self):
        self.client.get("/api/upcoming_races_stream?days=1")


class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 2)
