import { describe, expect, it } from "vitest";
import type { Joint, JointDofName } from "./types";
import { jointKindLabel, lockedDofs } from "./types";
import {
  computeArticulationPatch,
  computePartDeltas,
  currentValues,
  jointBasis,
  jointChainTo,
  partsInSubtree,
  poseFrom,
  solveIK,
  type JointPose,
} from "./articulation";
import { rigidApplyPoint } from "./rigid";

/** A screw joint with the given DoFs unlocked (range + optional value). */
const screw = (
  id: string,
  parent: string,
  child: string,
  pivot: { x: number; y: number; z: number },
  dofs: Partial<Record<JointDofName, { range: [number, number]; value?: number }>>,
  axis = { x: 0, y: 0, z: 1 },
): Joint => {
  const joint: Joint = {
    id,
    name: id,
    parentPartId: parent,
    childPartId: child,
    pivot,
    axis,
    dofs: lockedDofs(),
  };
  for (const [dof, spec] of Object.entries(dofs)) {
    joint.dofs[dof as JointDofName] = {
      range: spec.range,
      value: spec.value ?? 0,
    };
  }
  return joint;
};

const fullTwist = (value = 0) => ({
  twist: { range: [-Math.PI, Math.PI] as [number, number], value },
});

describe("computePartDeltas", () => {
  it("rotates a child part about the joint pivot", () => {
    const joints = [
      screw("j", "base", "arm", { x: 1, y: 0, z: 0 }, fullTwist(Math.PI / 2)),
    ];
    const deltas = computePartDeltas(joints);
    // a point at the pivot stays put; a point 1 unit +x of it swings to +y
    const atPivot = rigidApplyPoint(deltas.get("arm")!, { x: 1, y: 0, z: 0 });
    expect(atPivot.x).toBeCloseTo(1);
    expect(atPivot.y).toBeCloseTo(0);
    const tip = rigidApplyPoint(deltas.get("arm")!, { x: 2, y: 0, z: 0 });
    expect(tip.x).toBeCloseTo(1);
    expect(tip.y).toBeCloseTo(1);
    expect(tip.z).toBeCloseTo(0);
  });

  it("composes deltas down a chain", () => {
    const joints = [
      screw("j1", "base", "arm", { x: 0, y: 0, z: 0 }, fullTwist(Math.PI / 2)),
      screw("j2", "arm", "hand", { x: 1, y: 0, z: 0 }, fullTwist(Math.PI / 2)),
    ];
    // hand rest point (2,0,0): j2 swings it to (1,1,0) about (1,0,0), then
    // j1 rotates everything 90° about the origin → (-1,1,0)
    const deltas = computePartDeltas(joints);
    const tip = rigidApplyPoint(deltas.get("hand")!, { x: 2, y: 0, z: 0 });
    expect(tip.x).toBeCloseTo(-1);
    expect(tip.y).toBeCloseTo(1);
  });

  it("moves sliding children along the axis", () => {
    const joints = [
      screw(
        "j",
        "base",
        "slider",
        { x: 0, y: 0, z: 0 },
        { translation: { range: [-1, 1], value: 0.5 } },
        { x: 0, y: 1, z: 0 },
      ),
    ];
    const p = rigidApplyPoint(computePartDeltas(joints).get("slider")!, {
      x: 3,
      y: 0,
      z: 0,
    });
    expect(p.x).toBeCloseTo(3);
    expect(p.y).toBeCloseTo(0.5);
  });

  it("combines slide and twist on the same axis (a screw)", () => {
    const joints = [
      screw(
        "j",
        "base",
        "bolt",
        { x: 0, y: 0, z: 0 },
        {
          translation: { range: [-1, 1], value: 0.25 },
          twist: { range: [-Math.PI, Math.PI], value: Math.PI / 2 },
        },
      ),
    ];
    // (1,0,0) twists about z to (0,1,0), then slides +0.25 along z
    const p = rigidApplyPoint(computePartDeltas(joints).get("bolt")!, {
      x: 1,
      y: 0,
      z: 0,
    });
    expect(p.x).toBeCloseTo(0);
    expect(p.y).toBeCloseTo(1);
    expect(p.z).toBeCloseTo(0.25);
  });

  it("swings about the perpendicular reference axes", () => {
    const axis = { x: 0, y: 0, z: 1 };
    const { u } = jointBasis(axis);
    const joints = [
      screw(
        "j",
        "base",
        "ball",
        { x: 0, y: 0, z: 0 },
        { swingU: { range: [-Math.PI, Math.PI], value: Math.PI / 2 } },
        axis,
      ),
    ];
    // a point along the axis rotates 90° about u, landing perpendicular to
    // both u and the axis
    const p = rigidApplyPoint(computePartDeltas(joints).get("ball")!, {
      x: 0,
      y: 0,
      z: 1,
    });
    expect(p.x * u.x + p.y * u.y + p.z * u.z).toBeCloseTo(0);
    expect(p.z).toBeCloseTo(0);
    expect(Math.hypot(p.x, p.y, p.z)).toBeCloseTo(1);
  });
});

describe("jointBasis", () => {
  it("returns a right-handed orthonormal frame", () => {
    for (const axis of [
      { x: 0, y: 0, z: 1 },
      { x: 0, y: 1, z: 0 },
      { x: 0.6, y: 0.8, z: 0 },
    ]) {
      const { u, v } = jointBasis(axis);
      expect(Math.hypot(u.x, u.y, u.z)).toBeCloseTo(1);
      expect(Math.hypot(v.x, v.y, v.z)).toBeCloseTo(1);
      expect(u.x * axis.x + u.y * axis.y + u.z * axis.z).toBeCloseTo(0);
      expect(v.x * axis.x + v.y * axis.y + v.z * axis.z).toBeCloseTo(0);
      expect(u.x * v.x + u.y * v.y + u.z * v.z).toBeCloseTo(0);
    }
  });
});

describe("kind labels", () => {
  it("derives the effective joint kind from the unlocked DoFs", () => {
    const base = { x: 0, y: 0, z: 0 };
    expect(jointKindLabel(screw("j", "a", "b", base, {}))).toBe("fixed");
    expect(jointKindLabel(screw("j", "a", "b", base, fullTwist()))).toBe("revolute");
    expect(
      jointKindLabel(screw("j", "a", "b", base, { translation: { range: [0, 1] } })),
    ).toBe("prismatic");
    expect(
      jointKindLabel(
        screw("j", "a", "b", base, {
          translation: { range: [0, 1] },
          ...fullTwist(),
        }),
      ),
    ).toBe("cylindrical");
    expect(
      jointKindLabel(screw("j", "a", "b", base, { swingU: { range: [-1, 1] } })),
    ).toBe("ball");
  });
});

describe("tree helpers", () => {
  const joints = [
    screw("j1", "base", "arm", { x: 0, y: 0, z: 0 }, fullTwist()),
    screw("j2", "arm", "hand", { x: 1, y: 0, z: 0 }, fullTwist()),
    screw("j3", "base", "leg", { x: 0, y: 0, z: 0 }, fullTwist()),
  ];

  it("collects subtrees", () => {
    expect(partsInSubtree(joints, "arm")).toEqual(new Set(["arm", "hand"]));
    expect(partsInSubtree(joints, "base")).toEqual(
      new Set(["base", "arm", "hand", "leg"]),
    );
  });

  it("builds root-first chains", () => {
    expect(jointChainTo(joints, "hand").map((j) => j.id)).toEqual(["j1", "j2"]);
    expect(jointChainTo(joints, "base")).toEqual([]);
  });
});

describe("computeArticulationPatch", () => {
  it("is the exact inverse when going back to rest", () => {
    const joints = [
      screw("j", "base", "arm", { x: 1, y: 2, z: 0 }, fullTwist(0.7)),
    ];
    const patch = computeArticulationPatch(joints, currentValues, () => 0);
    // applying the patch to a posed point returns it to rest: pose the rest
    // point first, then patch it back
    const posed = rigidApplyPoint(computePartDeltas(joints).get("arm")!, {
      x: 5,
      y: 1,
      z: 2,
    });
    const back = rigidApplyPoint(patch.get("arm")!, posed);
    expect(back.x).toBeCloseTo(5);
    expect(back.y).toBeCloseTo(1);
    expect(back.z).toBeCloseTo(2);
  });
});

describe("solveIK", () => {
  const valueOf = (pose: JointPose, id: string, dof: JointDofName) =>
    pose.get(id)![dof];

  it("reaches a target with a two-link chain", () => {
    const joints = [
      screw("j1", "base", "arm", { x: 0, y: 0, z: 0 }, fullTwist()),
      screw("j2", "arm", "hand", { x: 1, y: 0, z: 0 }, fullTwist()),
    ];
    const chain = jointChainTo(joints, "hand");
    const pose = solveIK(
      joints,
      chain,
      poseFrom(joints),
      { x: 2, y: 0, z: 0 }, // end effector rest position (arm length 2)
      // interior of the reachable disc (full extension is a CCD singularity)
      { x: 1, y: 1, z: 0 },
    );
    const end = rigidApplyPoint(
      computePartDeltas(joints, (j, d) => valueOf(pose, j.id, d)).get("hand")!,
      { x: 2, y: 0, z: 0 },
    );
    expect(end.x).toBeCloseTo(1, 2);
    expect(end.y).toBeCloseTo(1, 2);
  });

  it("respects DoF ranges", () => {
    const joint = screw("j", "base", "arm", { x: 0, y: 0, z: 0 }, {
      twist: { range: [-0.5, 0.5] },
    });
    const pose = solveIK(
      [joint],
      [joint],
      poseFrom([joint]),
      { x: 1, y: 0, z: 0 },
      { x: -1, y: 0, z: 0 }, // wants π, gets the clamp
    );
    expect(valueOf(pose, "j", "twist")).toBeCloseTo(0.5);
  });

  it("uses sliding DoFs to move toward the target", () => {
    const joints = [
      screw(
        "j",
        "base",
        "slider",
        { x: 0, y: 0, z: 0 },
        { translation: { range: [-2, 2] } },
        { x: 1, y: 0, z: 0 },
      ),
    ];
    const pose = solveIK(
      joints,
      joints,
      poseFrom(joints),
      { x: 0, y: 1, z: 0 },
      { x: 1.5, y: 1, z: 0 },
    );
    expect(valueOf(pose, "j", "translation")).toBeCloseTo(1.5, 1);
  });

  it("uses swing DoFs when the twist alone cannot reach", () => {
    // axis +x, so the twist spins the arm about itself; only the swings can
    // lift the tip off the x-axis
    const joints = [
      screw(
        "j",
        "base",
        "arm",
        { x: 0, y: 0, z: 0 },
        {
          swingU: { range: [-Math.PI / 2, Math.PI / 2] },
          swingV: { range: [-Math.PI / 2, Math.PI / 2] },
        },
        { x: 1, y: 0, z: 0 },
      ),
    ];
    const pose = solveIK(
      joints,
      joints,
      poseFrom(joints),
      { x: 2, y: 0, z: 0 },
      { x: Math.SQRT2, y: Math.SQRT2, z: 0 }, // 45° up, same radius
    );
    const end = rigidApplyPoint(
      computePartDeltas(joints, (j, d) => valueOf(pose, j.id, d)).get("arm")!,
      { x: 2, y: 0, z: 0 },
    );
    expect(end.x).toBeCloseTo(Math.SQRT2, 1);
    expect(end.y).toBeCloseTo(Math.SQRT2, 1);
    expect(end.z).toBeCloseTo(0, 1);
  });
});
