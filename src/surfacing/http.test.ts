import { afterEach, describe, expect, it, vi } from "vitest";
import { request } from "./http";

function respond(
  body: string,
  init: { status?: number; type?: string } = {},
): void {
  const response = new Response(body, {
    status: init.status ?? 200,
    headers: { "content-type": init.type ?? "application/json" },
  });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));
}

/** The fallback a static host serves for an unknown path. */
const PAGE = "<!doctype html><html><body><div id=app></div></body></html>";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("request", () => {
  it("passes a normal response through", async () => {
    respond('{"ok":true}');
    expect(await (await request("/api/benchmark")).json()).toEqual({ ok: true });
  });

  // GitHub Pages and `vite preview` answer an unknown path with the app's own
  // index.html and a 200, so a successful fetch is not proof of a server —
  // without this the HTML reached JSON.parse, or was thrown as the error
  // message and shown to the user as a screenful of markup
  it("rejects an HTML page served with a 200", async () => {
    respond(PAGE, { type: "text/html" });
    await expect(request("/api/benchmark")).rejects.toThrow(
      /no surfacing server at this address/,
    );
  });

  it("never puts an HTML error page in the message", async () => {
    respond(PAGE, { status: 404, type: "text/html; charset=utf-8" });
    const error = await request("/api/benchmark").catch((exc: Error) => exc);
    expect((error as Error).message).not.toContain("<");
    expect((error as Error).message).toMatch(/no surfacing server at this address/);
  });

  // the server's own error text is worth showing, so it survives — the cap is
  // only there to stop an unbounded body filling an alert
  it("keeps a plain-text error body, on one line and bounded", async () => {
    respond("adapter failed:\n  no such method\n", { status: 400 });
    await expect(request("/api/benchmark")).rejects.toThrow(
      "400: adapter failed: no such method",
    );
  });

  it("truncates a long body", async () => {
    respond("x".repeat(5000), { status: 400 });
    const error = (await request("/api/benchmark").catch((exc: Error) => exc)) as Error;
    expect(error.message.length).toBeLessThan(340);
    expect(error.message.endsWith("…")).toBe(true);
  });

  it("falls back to the status when the body is empty", async () => {
    respond("", { status: 403 });
    await expect(request("/api/benchmark")).rejects.toThrow("403");
  });

  it("reports a dead server for a proxy 502 and for a network failure", async () => {
    respond("", { status: 502 });
    await expect(request("/api/benchmark")).rejects.toThrow(/unreachable/);

    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("failed")));
    await expect(request("/api/benchmark")).rejects.toThrow(/unreachable/);
  });
});
