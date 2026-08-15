// The one way this app talks to the surfacing server.
//
// Both halves of the API (client.ts for surfacing, benchmarkClient.ts for
// storage) had their own byte-identical copy of this; they share it now so a
// change to how failures are reported cannot apply to only one of them.

/** Fetch, with every way the server can be absent turned into one clear error.
 *
 *  The reason this is not just `fetch` is that "no server" has three different
 *  shapes and only one of them is a network error:
 *
 *  - nothing listening at all — fetch itself rejects;
 *  - the Vite dev proxy with nothing on the server port — a 5xx;
 *  - a static host (GitHub Pages, `vite preview`) — a *200* carrying the
 *    fallback HTML page, because an unknown path there is still a page.
 *
 *  The last one is the dangerous case: without the check below the HTML
 *  reaches `JSON.parse`, or gets thrown as an error message and shown to the
 *  user as a screenful of markup. */
export async function request(
  url: string,
  init?: RequestInit,
): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new Error(OFFLINE_HINT);
  }
  if (response.status === 500 || response.status === 502 || response.status === 504) {
    // what the Vite proxy returns when nothing listens on the server port
    throw new Error(OFFLINE_HINT);
  }
  if (!response.ok) throw new Error(await failureMessage(response));
  if (isHtml(response)) throw new Error(NO_SERVER_HINT);
  return response;
}

function isHtml(response: Response): boolean {
  return (response.headers.get("content-type") ?? "").includes("text/html");
}

/** What to report for a failed request. The body is the server's own error
 *  message and worth showing — but only when it is one: an HTML error page is
 *  noise, and any body at all can be long enough to fill the screen, so it is
 *  summarised rather than passed through. */
async function failureMessage(response: Response): Promise<string> {
  if (isHtml(response)) return NO_SERVER_HINT;
  const text = (await response.text().catch(() => "")).trim().replace(/\s+/g, " ");
  if (!text) return `${response.status} ${response.statusText}`;
  return text.length > 300
    ? `${response.status}: ${text.slice(0, 300)}…`
    : `${response.status}: ${text}`;
}

export const OFFLINE_HINT =
  "surfacing server unreachable — start it with " +
  "`uvicorn server:app --port 8801` in surfacing-server/ (see its README)";

const NO_SERVER_HINT =
  "no surfacing server at this address — it answered with a web page. " +
  "Use Open folder… in the benchmark window to view a saved benchmark " +
  "read-only, or start the server (see surfacing-server/README).";
