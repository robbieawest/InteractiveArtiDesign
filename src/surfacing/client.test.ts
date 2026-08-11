import { describe, expect, it } from "vitest";
import { viewSpecFor, type MethodInfo } from "./client";

const spec = { size: 518, count: 4, strokeColor: "#dcdcdc" };

function method(viewSpec: MethodInfo["viewSpec"]): MethodInfo {
  return { name: "m", params: [], viewSpec };
}

describe("viewSpecFor", () => {
  it("wants nothing for a method that consumes strokes as geometry", () => {
    expect(viewSpecFor(method(null), {})).toBeNull();
    expect(viewSpecFor({ name: "m", params: [] }, {})).toBeNull();
  });

  it("picks the spec the selecting option names", () => {
    const info = method({
      selector: "conditioner",
      specs: { views: spec },
    });
    expect(viewSpecFor(info, { conditioner: "views" })).toEqual(spec);
  });

  // a strategy that builds its input some other way contributes no entry, and
  // must render nothing rather than fall back to another strategy's spec
  it("wants nothing when the selected strategy declares no views", () => {
    const info = method({
      selector: "conditioner",
      specs: { views: spec },
    });
    expect(viewSpecFor(info, { conditioner: "voxels" })).toBeNull();
  });

  it("falls back to '*' when no option selects", () => {
    expect(viewSpecFor(method({ specs: { "*": spec } }), {})).toEqual(spec);
  });

  // options arrive from the panel as whatever type the param declared, and a
  // choice is a string — but a bool or int selector must not miss its key
  it("matches a non-string option value", () => {
    const info = method({ selector: "mode", specs: { "2": spec } });
    expect(viewSpecFor(info, { mode: 2 })).toEqual(spec);
  });

  describe("overrides", () => {
    const steerable = () =>
      method({
        specs: {
          "*": {
            ...spec,
            layout: "ring" as const,
            overrides: {
              views_count: "count" as const,
              views_layout: "layout" as const,
            },
          },
        },
      });

    it("folds the named options into the spec", () => {
      expect(
        viewSpecFor(steerable(), { views_count: 7, views_layout: "helix" }),
      ).toEqual({ ...spec, count: 7, layout: "helix" });
    });

    // the mapping is the method's business; the renderer takes a plain spec
    it("does not leak the mapping to the renderer", () => {
      const resolved = viewSpecFor(steerable(), { views_count: 7 });
      expect(resolved).not.toHaveProperty("overrides");
    });

    // an option the panel has not written yet must leave the method's own
    // default in place rather than blank the field
    it("keeps the declared value when the option is absent", () => {
      expect(viewSpecFor(steerable(), {})).toEqual({
        ...spec,
        layout: "ring",
      });
    });

    // a stale saved option, or a hand-written benchmark row, must not put an
    // unrenderable value in front of the renderer
    it("ignores values the field cannot take", () => {
      expect(
        viewSpecFor(steerable(), {
          views_count: Number.NaN,
          views_layout: "spiral",
        }),
      ).toEqual({ ...spec, layout: "ring" });
    });
  });
});
