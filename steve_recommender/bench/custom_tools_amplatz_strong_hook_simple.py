import eve

BASELINE_TIP_LENGTH_MM = 15.205308443374598
STRONG_HOOK_TIP_RADIUS_MM = 6.0


class JShapedAmplatzSuperStiffStrongHookSimple(eve.intervention.device.JShaped):
    def __init__(self):
        super().__init__(
            name="amplatz_super_stiff_jshaped_strong_hook",
            velocity_limit=(35.0, 3.14),
            length=450.0,
            tip_radius=STRONG_HOOK_TIP_RADIUS_MM,
            tip_angle=BASELINE_TIP_LENGTH_MM / STRONG_HOOK_TIP_RADIUS_MM,
            tip_outer_diameter=0.7,
            tip_inner_diameter=0.0,
            straight_outer_diameter=0.89,
            straight_inner_diameter=0.0,
            poisson_ratio=0.3,
            young_modulus_tip=424.0,
            young_modulus_straight=2436.0,
            mass_density_tip=2.1e-05,
            mass_density_straight=2.1e-05,
            visu_edges_per_mm=0.5,
            collis_edges_per_mm_tip=2.0,
            collis_edges_per_mm_straight=0.1,
            beams_per_mm_tip=1.4,
            beams_per_mm_straight=0.5,
            color=(0.0, 0.0, 0.0),
        )
