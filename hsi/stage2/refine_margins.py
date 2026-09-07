"""Current strict inner margins; publication thresholds are unchanged."""
def tail_hinge(values, allowed_depth):
    import torch
    if values.numel() == 0:
        raise ValueError("Cannot optimize an empty required SDF region")
    violations = torch.relu(-values - float(allowed_depth))
    top = torch.topk(violations, min(12, values.numel())).values
    return violations.max().square() + top.square().mean()

def strict_terms(vertices,fields,anatomy,thresholds,grid_term):
    target,outside=fields.target.sample(vertices)
    obstacle,obstacle_outside=fields.non_target.sample(vertices)
    allowed=anatomy.allowed_contact.to(vertices.device).bool()
    margin=.005
    allowed_loss=tail_hinge(target[allowed & ~outside],
        max(0.,float(thresholds["maximum_target_allowed_contact_penetration_m"])-margin))
    forbidden_loss=tail_hinge(target[~allowed & ~outside],
        max(0.,float(thresholds["maximum_target_forbidden_body_penetration_m"])-margin))
    obstacle_loss=tail_hinge(obstacle[~obstacle_outside],
        max(0.,float(thresholds["maximum_non_target_scene_penetration_m"])-margin))
    boundary,_=grid_term(vertices)
    return 3500*allowed_loss+1500*forbidden_loss+1500*obstacle_loss+150*boundary

def bilateral_support_loss(vertices,anatomy,thresholds,band):
    import torch
    result=vertices.new_zeros(())
    for side in ("left","right"):
        mask=getattr(anatomy,side+"_foot_support")
        if mask is None:
            raise ValueError("Independent bilateral foot masks are required")
        values=vertices[mask.to(vertices.device).bool(),2].abs()
        required=int(thresholds["minimum_"+side+"_foot_ground_vertices"])
        if required<1 or values.numel()<required:
            raise ValueError("Invalid required foot contact area")
        closest=torch.topk(values,required,largest=False).values
        # A strict inner optimization margin; the actual 30-mm contact band is unchanged.
        inner=max(0.,float(band)-.008)
        result=result+1500*torch.relu(closest.max()-inner).square()+200*closest.square().mean()
    return result

def violation_rank(metrics,gates,thresholds):
    """Do not prefer prettier-but-more-penetrating poses when both fail physics."""
    severity=0.
    for metric,limit in (("ground_penetration_m","maximum_ground_penetration_m"),
            ("non_target_scene_penetration_m","maximum_non_target_scene_penetration_m"),
            ("target_allowed_contact_penetration_m","maximum_target_allowed_contact_penetration_m"),
            ("target_forbidden_body_penetration_m","maximum_target_forbidden_body_penetration_m")):
        severity+=max(0.,float(metrics[metric])/max(1e-8,float(thresholds[limit]))-1.)
    for side in ("left","right"):
        required=int(thresholds["minimum_"+side+"_foot_ground_vertices"])
        severity+=max(0.,1.-float(metrics[side+"_foot_ground_vertex_count_in_band"])/required)
        severity+=max(0.,float(metrics[side+"_foot_ground_surface_topk_mean_m"])/.030-1.)
    return (sum(not bool(v) for v in gates.values()),severity)
