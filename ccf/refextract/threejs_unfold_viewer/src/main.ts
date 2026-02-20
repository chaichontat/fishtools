import * as THREE from "three";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls.js";

import { loadAssets } from "./io";
import { applyUnfold, createUnfoldContext } from "./unfold";

declare const __VIEWER_DIR_ABS__: string;
declare const __REPO_ROOT_ABS__: string;

const DEFAULT_CAMERA_YAW_DEG = 30.0;
const DEFAULT_CAMERA_PITCH_DEG = -80.0;
const DEFAULT_CAMERA_ROLL_DEG = -150.0;
const DEFAULT_CAMERA_DIST = 12000.0;

function shadeNeoSupportColors(
  baseColors: Uint8Array,
  neoTSupport: Uint8Array | null,
  nVertices: number,
): Uint8Array {
  if (baseColors.length !== nVertices * 3) {
    throw new Error(
      `colors length mismatch: ${baseColors.length} vs ${nVertices * 3}`,
    );
  }
  if (neoTSupport === null) {
    return baseColors;
  }
  if (neoTSupport.length !== nVertices) {
    throw new Error(
      `neo_t_support length mismatch: ${neoTSupport.length} vs ${nVertices}`,
    );
  }

  const out = new Uint8Array(baseColors.length);
  out.set(baseColors);
  for (let v = 0; v < nVertices; v += 1) {
    const i = 3 * v;
    const r = out[i + 0];
    const g = out[i + 1];
    const b = out[i + 2];
    if (neoTSupport[v] !== 0) {
      // Keep support area at the original AP/ML gradient colors.
      out[i + 0] = r;
      out[i + 1] = g;
      out[i + 2] = b;
      continue;
    }
    // Aggressively de-emphasize non-support area.
    const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    const lightGray = Math.min(255, Math.round(0.6 * gray + 30.0));
    out[i + 0] = lightGray;
    out[i + 1] = lightGray;
    out[i + 2] = lightGray;
  }
  return out;
}

function createDirectionSprite(
  text: string,
  colorHex: string,
): { sprite: THREE.Sprite; aspect: number } {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Failed to create 2D canvas context for direction label.");
  }

  const fontPx = 44;
  const padX = 20;
  const padY = 14;
  context.font = `600 ${fontPx}px "IBM Plex Sans", "Avenir Next", sans-serif`;
  const textW = Math.ceil(context.measureText(text).width);
  canvas.width = textW + 2 * padX;
  canvas.height = fontPx + 2 * padY;

  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "rgba(247,249,239,0.94)";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = "rgba(72,76,64,0.65)";
  context.lineWidth = 3;
  context.strokeRect(1.5, 1.5, canvas.width - 3, canvas.height - 3);
  context.font = `600 ${fontPx}px "IBM Plex Sans", "Avenir Next", sans-serif`;
  context.fillStyle = colorHex;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, canvas.width / 2, canvas.height / 2 + 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: true,
    depthWrite: true,
    alphaTest: 0.02,
  });
  const sprite = new THREE.Sprite(material);
  return { sprite, aspect: canvas.width / canvas.height };
}

function updateDirectionSprites({
  geometry,
  labels,
}: {
  geometry: THREE.BufferGeometry;
  labels: {
    anterior: { sprite: THREE.Sprite; aspect: number };
    posterior: { sprite: THREE.Sprite; aspect: number };
    lateral: { sprite: THREE.Sprite; aspect: number };
    medial: { sprite: THREE.Sprite; aspect: number };
  };
}): void {
  const positionAttr = geometry.getAttribute(
    "position",
  ) as THREE.BufferAttribute;
  const bbox = new THREE.Box3().setFromBufferAttribute(positionAttr);
  const center = bbox.getCenter(new THREE.Vector3());
  const size = bbox.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.z, 1.0);
  const apPad = 0.45 * extent;
  const y = center.y;
  const textH = Math.max(34.0, 0.035 * Math.max(size.length(), 1.0));
  const mlPad = Math.max(0.35 * extent, 1.8 * textH);

  // AP labels are anchored at the AP axis line endpoints.
  labels.anterior.sprite.position.set(center.x, y, bbox.min.z - apPad);
  labels.posterior.sprite.position.set(center.x, y, bbox.max.z + apPad);
  labels.lateral.sprite.position.set(bbox.min.x - mlPad, y, center.z);
  labels.medial.sprite.position.set(bbox.max.x + mlPad, y, center.z);

  labels.anterior.sprite.scale.set(textH * labels.anterior.aspect, textH, 1.0);
  labels.posterior.sprite.scale.set(
    textH * labels.posterior.aspect,
    textH,
    1.0,
  );
  labels.lateral.sprite.scale.set(textH * labels.lateral.aspect, textH, 1.0);
  labels.medial.sprite.scale.set(textH * labels.medial.aspect, textH, 1.0);
}

function updateApAxisLine({
  geometry,
  line,
}: {
  geometry: THREE.BufferGeometry;
  line: THREE.Line;
}): void {
  const positionAttr = geometry.getAttribute(
    "position",
  ) as THREE.BufferAttribute;
  const bbox = new THREE.Box3().setFromBufferAttribute(positionAttr);
  const center = bbox.getCenter(new THREE.Vector3());
  const size = bbox.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.z, 1.0);
  const pad = 0.5 * extent;

  const linePos = (line.geometry.getAttribute("position") ??
    null) as THREE.BufferAttribute | null;
  if (!linePos || linePos.count < 2) {
    throw new Error("AP axis line geometry is not initialized.");
  }
  linePos.setXYZ(0, center.x, center.y, bbox.min.z - pad);
  linePos.setXYZ(1, center.x, center.y, bbox.max.z + pad);
  linePos.needsUpdate = true;
  line.geometry.computeBoundingSphere();
}

function updateMlApPlane({
  geometry,
  planeFill,
  planeOutline,
}: {
  geometry: THREE.BufferGeometry;
  planeFill: THREE.Mesh;
  planeOutline: THREE.LineSegments;
}): void {
  const positionAttr = geometry.getAttribute(
    "position",
  ) as THREE.BufferAttribute;
  const bbox = new THREE.Box3().setFromBufferAttribute(positionAttr);
  const center = bbox.getCenter(new THREE.Vector3());
  const size = bbox.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.z, 1.0);
  const pad = 0.15 * extent;
  const width = Math.max(1.0, size.x + 2.0 * pad);
  const depth = Math.max(1.0, size.z + 2.0 * pad);

  planeFill.position.set(center.x, center.y, center.z);
  planeFill.scale.set(width, depth, 1.0);

  planeOutline.position.set(center.x, center.y, center.z);
  planeOutline.scale.set(width, depth, 1.0);
}

function wrapDegSigned(deg: number): number {
  return ((((deg + 180.0) % 360.0) + 360.0) % 360.0) - 180.0;
}

function updateCameraReadout({
  camera,
  controls,
  elem,
}: {
  camera: THREE.PerspectiveCamera;
  controls: TrackballControls;
  elem: HTMLDivElement;
}): void {
  const offset = new THREE.Vector3().subVectors(
    camera.position,
    controls.target,
  );
  const dist = offset.length();
  const azimuthY = Math.atan2(offset.x, offset.z) * (180.0 / Math.PI);
  const elevX =
    Math.atan2(offset.y, Math.hypot(offset.x, offset.z)) * (180.0 / Math.PI);
  const euler = new THREE.Euler().setFromQuaternion(camera.quaternion, "YXZ");
  const rollZ = euler.z * (180.0 / Math.PI);
  elem.textContent = `camera yaw(Y)=${wrapDegSigned(azimuthY).toFixed(1)}° pitch(X)=${wrapDegSigned(elevX).toFixed(1)}° roll(Z)=${wrapDegSigned(rollZ).toFixed(1)}° dist=${dist.toFixed(1)}`;
}

function setCameraPose({
  camera,
  controls,
  target,
  yawDeg,
  pitchDeg,
  rollDeg,
  dist,
}: {
  camera: THREE.PerspectiveCamera;
  controls: TrackballControls;
  target: THREE.Vector3;
  yawDeg: number;
  pitchDeg: number;
  rollDeg: number;
  dist: number;
}): void {
  const yaw = (yawDeg * Math.PI) / 180.0;
  const pitch = (pitchDeg * Math.PI) / 180.0;
  const roll = (rollDeg * Math.PI) / 180.0;
  const cp = Math.cos(pitch);
  const offset = new THREE.Vector3(
    dist * Math.sin(yaw) * cp,
    dist * Math.sin(pitch),
    dist * Math.cos(yaw) * cp,
  );
  const position = new THREE.Vector3().copy(target).add(offset);
  camera.position.copy(position);

  const forward = new THREE.Vector3().subVectors(target, position).normalize();
  let worldUp = new THREE.Vector3(0, 1, 0);
  if (Math.abs(forward.dot(worldUp)) > 0.98) {
    worldUp = new THREE.Vector3(0, 0, 1);
  }
  const right = new THREE.Vector3().crossVectors(forward, worldUp).normalize();
  const up0 = new THREE.Vector3().crossVectors(right, forward).normalize();
  const up = up0
    .clone()
    .applyQuaternion(new THREE.Quaternion().setFromAxisAngle(forward, roll));
  camera.up.copy(up);
  camera.lookAt(target);

  controls.target.copy(target);
  controls.update();
}

function formatScaleLabelUm(lenUm: number): string {
  if (lenUm >= 1000.0) {
    const mm = lenUm / 1000.0;
    const decimals = mm >= 10.0 ? 0 : 1;
    return `${mm.toFixed(decimals)} mm`;
  }
  return `${Math.round(lenUm)} um`;
}

function updateScaleBar({
  camera,
  controls,
  renderer,
  barElem,
  labelElem,
}: {
  camera: THREE.PerspectiveCamera;
  controls: TrackballControls;
  renderer: THREE.WebGLRenderer;
  barElem: HTMLDivElement;
  labelElem: HTMLDivElement;
}): void {
  const size = renderer.getSize(new THREE.Vector2());
  const widthPx = Math.max(1.0, size.x);
  const right = new THREE.Vector3();
  const up = new THREE.Vector3();
  const forward = new THREE.Vector3();
  camera.matrixWorld.extractBasis(right, up, forward);
  right.normalize();
  const center = controls.target.clone();
  const probeUm = 1000.0;
  const p0 = center.clone();
  const p1 = center.clone().add(right.multiplyScalar(probeUm));
  p0.project(camera);
  p1.project(camera);
  const pxPerUm = (Math.abs(p1.x - p0.x) * 0.5 * widthPx) / probeUm;
  if (!Number.isFinite(pxPerUm) || pxPerUm <= 1.0e-6) {
    barElem.style.width = "0px";
    labelElem.textContent = "";
    return;
  }

  const candidatesUm = [100, 200, 500, 1000, 2000, 5000, 10000, 20000];
  const minPx = 80.0;
  const maxPx = 220.0;
  const targetPx = 140.0;
  let bestUm = candidatesUm[0];
  let bestScore = Number.POSITIVE_INFINITY;
  for (const c of candidatesUm) {
    const px = c * pxPerUm;
    let score = Math.abs(px - targetPx);
    if (px < minPx) {
      score += (minPx - px) * 2.0;
    }
    if (px > maxPx) {
      score += (px - maxPx) * 2.0;
    }
    if (score < bestScore) {
      bestScore = score;
      bestUm = c;
    }
  }
  const barPx = Math.max(24.0, bestUm * pxPerUm);
  barElem.style.width = `${barPx.toFixed(1)}px`;
  labelElem.textContent = formatScaleLabelUm(bestUm);
}

function getElement<T extends HTMLElement>(id: string): T {
  const elem = document.getElementById(id);
  if (!elem) {
    throw new Error(`Missing element #${id}`);
  }
  return elem as T;
}

function parseAssetsBase(): string {
  const url = new URL(window.location.href);
  const queryValue = url.searchParams.get("assets");
  const raw =
    queryValue && queryValue.trim().length > 0
      ? queryValue.trim()
      : "../../out/refextract/midsurface_neocortex_mesocortex_allocortex_3d/threejs_unfold";

  if (
    raw.startsWith("http://") ||
    raw.startsWith("https://") ||
    raw.startsWith("/@fs/")
  ) {
    return raw;
  }
  if (raw.startsWith("/")) {
    return raw;
  }

  // Resolve relative filesystem paths through Vite's /@fs route.
  const viewerFileUrl = `file://${__VIEWER_DIR_ABS__.endsWith("/") ? __VIEWER_DIR_ABS__ : `${__VIEWER_DIR_ABS__}/`}`;
  const resolved = new URL(raw, viewerFileUrl);
  if (resolved.protocol !== "file:") {
    throw new Error(`Unsupported assets path '${raw}'.`);
  }
  return `/@fs${resolved.pathname}`;
}

async function main(): Promise<void> {
  const canvasRoot = getElement<HTMLDivElement>("canvas-root");
  const progressInput = getElement<HTMLInputElement>("progress");
  const playButton = getElement<HTMLButtonElement>("play");
  const resetButton = getElement<HTMLButtonElement>("reset");
  const flipInput = getElement<HTMLInputElement>("flip");
  const speedSelect = getElement<HTMLSelectElement>("speed");
  const durationInput = getElement<HTMLInputElement>("duration");
  const statusElem = getElement<HTMLDivElement>("status");
  const cameraReadoutElem = getElement<HTMLDivElement>("camera-readout");
  const scaleBarElem = getElement<HTMLDivElement>("scalebar");
  const scaleBarLabelElem = getElement<HTMLDivElement>("scalebar-label");
  const errorElem = getElement<HTMLDivElement>("error");

  const assetsBase = parseAssetsBase();
  statusElem.textContent = `Loading assets from: ${assetsBase} (repo: ${__REPO_ROOT_ABS__})`;
  const loaded = await loadAssets(assetsBase);
  const phase1Frac = loaded.manifest.phase1_frac_default;
  const shadedColors = shadeNeoSupportColors(
    loaded.colors,
    loaded.neoTSupport,
    loaded.manifest.n_vertices,
  );
  const context = createUnfoldContext(
    loaded.manifest,
    loaded.positions,
    loaded.segLen,
    loaded.theta,
    loaded.ap,
  );

  const geometry = new THREE.BufferGeometry();
  const dynamicPositions = new Float32Array(loaded.positions.length);
  dynamicPositions.set(loaded.positions);
  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(dynamicPositions, 3),
  );
  geometry.setIndex(new THREE.BufferAttribute(loaded.faces, 1));
  geometry.setAttribute(
    "color",
    new THREE.BufferAttribute(shadedColors, 3, true),
  );
  geometry.computeVertexNormals();

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.58,
    metalness: 0.03,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geometry, material);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#f4f7ec");
  scene.add(mesh);
  const labels = {
    anterior: createDirectionSprite("Anterior (-Z)", "#93354e"),
    posterior: createDirectionSprite("Posterior (+Z)", "#93354e"),
    lateral: createDirectionSprite("Lateral (-X)", "#245f8f"),
    medial: createDirectionSprite("Medial (+X)", "#245f8f"),
  };
  scene.add(labels.anterior.sprite);
  scene.add(labels.posterior.sprite);
  scene.add(labels.lateral.sprite);
  scene.add(labels.medial.sprite);
  const mlApPlaneGeometry = new THREE.PlaneGeometry(1, 1);
  const mlApPlane = new THREE.Mesh(
    mlApPlaneGeometry,
    new THREE.MeshBasicMaterial({
      color: 0x1f3b5c,
      transparent: true,
      opacity: 0.13,
      side: THREE.DoubleSide,
      depthTest: true,
      depthWrite: false,
    }),
  );
  mlApPlane.rotation.x = -Math.PI / 2.0;
  scene.add(mlApPlane);
  const mlApPlaneOutline = new THREE.LineSegments(
    new THREE.EdgesGeometry(mlApPlaneGeometry),
    new THREE.LineBasicMaterial({
      color: 0x1f3b5c,
      transparent: true,
      opacity: 0.7,
      depthTest: true,
      depthWrite: false,
    }),
  );
  mlApPlaneOutline.rotation.x = -Math.PI / 2.0;
  scene.add(mlApPlaneOutline);
  const apAxisLineGeometry = new THREE.BufferGeometry();
  apAxisLineGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(6), 3),
  );
  const apAxisLine = new THREE.Line(
    apAxisLineGeometry,
    new THREE.LineBasicMaterial({
      color: 0x8b1c1c,
      transparent: true,
      opacity: 0.9,
      depthTest: true,
      depthWrite: false,
    }),
  );
  scene.add(apAxisLine);

  const hemi = new THREE.HemisphereLight(0xfaf9ef, 0xb9c7a0, 0.85);
  scene.add(hemi);
  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(2.0, 1.4, 1.1);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xc8d7ff, 0.45);
  rim.position.set(-1.4, 0.7, -1.3);
  scene.add(rim);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(canvasRoot.clientWidth, canvasRoot.clientHeight);
  canvasRoot.appendChild(renderer.domElement);

  const camera = new THREE.PerspectiveCamera(
    40,
    canvasRoot.clientWidth / canvasRoot.clientHeight,
    0.1,
    1e7,
  );
  const bbox = new THREE.Box3().setFromBufferAttribute(
    geometry.getAttribute("position") as THREE.BufferAttribute,
  );
  const center = bbox.getCenter(new THREE.Vector3());

  const controls = new TrackballControls(camera, renderer.domElement);
  controls.rotateSpeed = 4.0;
  controls.panSpeed = 0.9;
  controls.zoomSpeed = 1.2;
  controls.dynamicDampingFactor = 0.1;
  controls.staticMoving = false;
  setCameraPose({
    camera,
    controls,
    target: center,
    yawDeg: DEFAULT_CAMERA_YAW_DEG,
    pitchDeg: DEFAULT_CAMERA_PITCH_DEG,
    rollDeg: DEFAULT_CAMERA_ROLL_DEG,
    dist: DEFAULT_CAMERA_DIST,
  });

  window.addEventListener("resize", () => {
    const w = canvasRoot.clientWidth;
    const h = canvasRoot.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    controls.handleResize();
  });

  let progress = Number(progressInput.value);
  let playing = false;
  let lastTimeSec = performance.now() * 0.001;

  const applyCurrentState = (): void => {
    const flipXZ = flipInput.checked;
    applyUnfold(context, {
      progress,
      phase1Frac,
      flipXZ,
      outXYZ: dynamicPositions,
    });
    const positionAttr = geometry.getAttribute(
      "position",
    ) as THREE.BufferAttribute;
    positionAttr.needsUpdate = true;
    geometry.computeVertexNormals();
    updateDirectionSprites({ geometry, labels });
    updateApAxisLine({ geometry, line: apAxisLine });
    updateMlApPlane({
      geometry,
      planeFill: mlApPlane,
      planeOutline: mlApPlaneOutline,
    });
    statusElem.textContent = `progress=${progress.toFixed(3)} flipXZ=${flipXZ}`;
  };

  const setPlaying = (value: boolean): void => {
    playing = value;
    playButton.textContent = playing ? "Pause" : "Play";
  };

  progressInput.addEventListener("input", () => {
    progress = Number(progressInput.value);
    setPlaying(false);
    applyCurrentState();
  });
  flipInput.addEventListener("change", applyCurrentState);
  playButton.addEventListener("click", () => setPlaying(!playing));
  resetButton.addEventListener("click", () => {
    setCameraPose({
      camera,
      controls,
      target: center,
      yawDeg: DEFAULT_CAMERA_YAW_DEG,
      pitchDeg: DEFAULT_CAMERA_PITCH_DEG,
      rollDeg: DEFAULT_CAMERA_ROLL_DEG,
      dist: DEFAULT_CAMERA_DIST,
    });
  });

  applyCurrentState();

  const tick = (): void => {
    const nowSec = performance.now() * 0.001;
    const dt = Math.max(0.0, nowSec - lastTimeSec);
    lastTimeSec = nowSec;

    if (playing) {
      const speed = Number(speedSelect.value);
      const duration = Math.max(1.0, Number(durationInput.value));
      progress = Math.min(1.0, progress + (dt * speed) / duration);
      progressInput.value = progress.toFixed(3);
      if (progress >= 1.0) {
        setPlaying(false);
      }
      applyCurrentState();
    }

    controls.update();
    updateCameraReadout({ camera, controls, elem: cameraReadoutElem });
    updateScaleBar({
      camera,
      controls,
      renderer,
      barElem: scaleBarElem,
      labelElem: scaleBarLabelElem,
    });
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  };

  requestAnimationFrame(tick);
}

void main().catch((err: unknown) => {
  const errorElem = document.getElementById("error");
  if (errorElem) {
    errorElem.textContent =
      err instanceof Error ? (err.stack ?? err.message) : String(err);
  }
  throw err;
});
