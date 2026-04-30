from steve_recommender.train_v2.runtime.execution import compute_resume_targets


def test_compute_resume_targets_keeps_requested_absolute_target_when_larger():
    heatup_steps, training_target = compute_resume_targets(
        requested_heatup_steps=100,
        requested_training_steps=500,
        current_explore_steps=300,
        resume_skip_heatup=False,
    )
    assert heatup_steps == 100
    assert training_target == 500


def test_compute_resume_targets_treats_smaller_training_steps_as_additional():
    heatup_steps, training_target = compute_resume_targets(
        requested_heatup_steps=100,
        requested_training_steps=200,
        current_explore_steps=300,
        resume_skip_heatup=False,
    )
    assert heatup_steps == 100
    assert training_target == 500


def test_compute_resume_targets_can_skip_heatup():
    heatup_steps, training_target = compute_resume_targets(
        requested_heatup_steps=100,
        requested_training_steps=200,
        current_explore_steps=300,
        resume_skip_heatup=True,
    )
    assert heatup_steps == 0
    assert training_target == 500
