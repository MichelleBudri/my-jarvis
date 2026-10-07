// Copy text from the HUD. In the native window WebKit may refuse the Clipboard API, so
// the window's own `copy` is tried first, then the browser's, then the old selection way.

type PyWebView = { api?: { copy?: (text: string) => Promise<boolean> } };

export async function copyText(text: string): Promise<boolean> {
  const native = (window as unknown as { pywebview?: PyWebView }).pywebview?.api;
  if (native?.copy) {
    try {
      if (await native.copy(text)) return true;
    } catch {
      // fall through to the browser's ways
    }
  }
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    return ok;
  }
}
