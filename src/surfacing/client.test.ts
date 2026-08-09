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
});
