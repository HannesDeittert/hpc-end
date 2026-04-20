import eve

BASELINE_TIP_LENGTH_MM = 15.205308443374598
STRAIGHT_TIP_RADIUS_MM = 1000.0


class JShapedUniversalIIStraightSimple(eve.intervention.device.JShaped):
    def __init__(self):
        super().__init__(
            name="abbott_universal_II_jshaped_straight",
            velocity_limit=(35.0, 3.14),
            length=450.0,
            tip_radius=STRAIGHT_TIP_RADIUS_MM,
            tip_angle=BASELINE_TIP_LENGTH_MM / STRAIGHT_TIP_RADIUS_MM,
            tip_outer_diameter=0.36,
            tip_inner_diameter=0.0,
            straight_outer_diameter=0.36,
            straight_inner_diameter=0.0,
            poisson_ratio=0.33,
            young_modulus_tip=1800.0,
            young_modulus_straight=8500.0,
            mass_density_tip=2.1e-05,
            mass_density_straight=2.1e-05,
            visu_edges_per_mm=0.5,
            collis_edges_per_mm_tip=2.0,
            collis_edges_per_mm_straight=0.1,
            beams_per_mm_tip=1.4,
            beams_per_mm_straight=0.5,
            color=(0.0, 0.0, 0.0),
        )
