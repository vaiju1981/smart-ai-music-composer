/**
 * Browser-side bridge between the Node driver and OpenSheetMusicDisplay.
 *
 * The driver (src/render.ts) navigates to viewer.html, evaluates
 * `renderMusicXml(xml)`, and takes the SVG (returned markup) or a PNG
 * (element screenshot) from the container. Plain JS: this file runs in
 * the browser, where OSMD's UMD build exposes `OpenSheetMusicDisplay`.
 */
/* global OpenSheetMusicDisplay */

window.renderMusicXml = async function renderMusicXml(xml) {
  const container = document.getElementById("osmd-container");
  container.innerHTML = "";
  const osmd = new OpenSheetMusicDisplay(container, {
    backend: "svg",
    disableCursor: true,
  });
  await osmd.load(xml);
  osmd.render();
  return container.innerHTML;
};