import * as THREE from "three";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls.js";

import { loadAssets } from "./io";
import { applyUnfold, createUnfoldContext } from "./unfold";

declare const __VIEWER_DIR_ABS__: string;
declare const __REPO_ROOT_ABS__: string;

const DEFAULT_CAMERA_YAW_DEG = -180.0;
const DEFAULT_CAMERA_PITCH_DEG = -90.0;
const DEFAULT_CAMERA_ROLL_DEG = 180.0;
const DEFAULT_CAMERA_DIST = 12000.0;
const PLAYBACK_DURATION_SEC = 4.0;
const HEMISPHERE_TRANSITION_SEC = 0.85;
const OVERLAY_STAGE_START_FRAC = 0.12;
const OVERLAY_STAGE_END_FRAC = 0.68;
const BRAIN_STAGE_START_FRAC = 0.78;
const HEMISPHERE_RIGHT_SLIDE_FRAC = 0.0;
const HEMISPHERE_LEFT_ENTRY_FRAC = 0.0;
const PLANE_FILL_OPACITY = 0.13;
const PLANE_OUTLINE_OPACITY = 0.7;
const AP_AXIS_OPACITY = 0.9;
const CAMERA_LIGHT_ELEVATION_DEG = 30.0;
const CAMERA_LIGHT_INTENSITY = 0.42;
const PLANE_OFFSET_UM = 8.0;

type HemisphereSide = "left" | "right";

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

function createDirectionSpriteTexture(
  text: string,
  colorHex: string,
): { texture: THREE.CanvasTexture; aspect: number } {
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
  return { texture, aspect: canvas.width / canvas.height };
}

function createDirectionSprite(
  text: string,
  colorHex: string,
): { sprite: THREE.Sprite; aspect: number } {
  const { texture, aspect } = createDirectionSpriteTexture(text, colorHex);
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: true,
    depthWrite: true,
    alphaTest: 0.02,
  });
  const sprite = new THREE.Sprite(material);
  return { sprite, aspect };
}

function setDirectionSpriteText(
  label: { sprite: THREE.Sprite; aspect: number },
  text: string,
  colorHex: string,
): void {
  const { texture, aspect } = createDirectionSpriteTexture(text, colorHex);
  const material = label.sprite.material as THREE.SpriteMaterial;
  material.map?.dispose();
  material.map = texture;
  material.needsUpdate = true;
  label.aspect = aspect;
}

function updateDirectionSprites({
  bounds,
  labels,
}: {
  bounds: THREE.Box3;
  labels: {
    anterior: { sprite: THREE.Sprite; aspect: number };
    posterior: { sprite: THREE.Sprite; aspect: number };
    negativeX: { sprite: THREE.Sprite; aspect: number };
    positiveX: { sprite: THREE.Sprite; aspect: number };
  };
}): void {
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.z, 1.0);
  const apPad = 0.45 * extent;
  const y = center.y;
  const textH = Math.max(34.0, 0.035 * Math.max(size.length(), 1.0));
  const mlPad = Math.max(0.35 * extent, 1.8 * textH);

  // AP labels are anchored at the AP axis line endpoints.
  labels.anterior.sprite.position.set(center.x, y, bounds.min.z - apPad);
  labels.posterior.sprite.position.set(center.x, y, bounds.max.z + apPad);
  labels.negativeX.sprite.position.set(bounds.min.x - mlPad, y, center.z);
  labels.positiveX.sprite.position.set(bounds.max.x + mlPad, y, center.z);

  labels.anterior.sprite.scale.set(textH * labels.anterior.aspect, textH, 1.0);
  labels.posterior.sprite.scale.set(
    textH * labels.posterior.aspect,
    textH,
    1.0,
  );
  labels.negativeX.sprite.scale.set(textH * labels.negativeX.aspect, textH, 1.0);
  labels.positiveX.sprite.scale.set(textH * labels.positiveX.aspect, textH, 1.0);
}

function updateApAxisLine({
  bounds,
  line,
}: {
  bounds: THREE.Box3;
  line: THREE.Line;
}): void {
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.z, 1.0);
  const pad = 0.5 * extent;

  const linePos = (line.geometry.getAttribute("position") ??
    null) as THREE.BufferAttribute | null;
  if (!linePos || linePos.count < 2) {
    throw new Error("AP axis line geometry is not initialized.");
  }
  linePos.setXYZ(0, center.x, center.y, bounds.min.z - pad);
  linePos.setXYZ(1, center.x, center.y, bounds.max.z + pad);
  linePos.needsUpdate = true;
  line.geometry.computeBoundingSphere();
}

function updateMlApPlane({
  bounds,
  planeFill,
  planeOutline,
}: {
  bounds: THREE.Box3;
  planeFill: THREE.Mesh;
  planeOutline: THREE.LineSegments;
}): void {
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.z, 1.0);
  const pad = 0.15 * extent;
  const width = Math.max(1.0, size.x + 2.0 * pad);
  const depth = Math.max(1.0, size.z + 2.0 * pad);
  const offsetY = center.y - PLANE_OFFSET_UM;

  planeFill.position.set(center.x, offsetY, center.z);
  planeFill.scale.set(width, depth, 1.0);

  planeOutline.position.set(center.x, offsetY, center.z);
  planeOutline.scale.set(width, depth, 1.0);
}

function inferSourceHemisphere(
  positions: Float32Array,
  mirrorPlaneX: number,
): HemisphereSide {
  let sumX = 0.0;
  let count = 0;
  for (let idx = 0; idx < positions.length; idx += 3) {
    sumX += positions[idx];
    count += 1;
  }
  if (count === 0) {
    throw new Error("Cannot infer hemisphere from an empty positions array.");
  }
  return sumX / count < mirrorPlaneX ? "left" : "right";
}

function getMlAxisLabels({
  sourceHemisphere,
  showBoth,
}: {
  sourceHemisphere: HemisphereSide;
  showBoth: boolean;
}): { negativeX: string; positiveX: string; legendLine: string } {
  if (showBoth) {
    return {
      negativeX: "Lateral (-X)",
      positiveX: "Lateral (+X)",
      legendLine: "Lateral on both hemispheres",
    };
  }
  if (sourceHemisphere === "left") {
    return {
      negativeX: "Lateral (-X)",
      positiveX: "Medial (+X)",
      legendLine: "Lateral: -X, Medial: +X",
    };
  }
  return {
    negativeX: "Medial (-X)",
    positiveX: "Lateral (+X)",
    legendLine: "Medial: -X, Lateral: +X",
  };
}

function computeActiveBounds({
  bounds,
  rightTranslateX,
  mirroredPositionX,
  blendFactor,
}: {
  bounds: THREE.Box3;
  rightTranslateX: number;
  mirroredPositionX: number;
  blendFactor: number;
}): THREE.Box3 {
  const rightBounds = new THREE.Box3(
    new THREE.Vector3(
      bounds.min.x + rightTranslateX,
      bounds.min.y,
      bounds.min.z,
    ),
    new THREE.Vector3(
      bounds.max.x + rightTranslateX,
      bounds.max.y,
      bounds.max.z,
    ),
  );
  if (blendFactor <= 1.0e-6) {
    return rightBounds;
  }
  const mirroredMinX = Math.min(
    (-bounds.min.x) + mirroredPositionX,
    (-bounds.max.x) + mirroredPositionX,
  );
  const mirroredMaxX = Math.max(
    (-bounds.min.x) + mirroredPositionX,
    (-bounds.max.x) + mirroredPositionX,
  );
  const bothBounds = rightBounds.clone().union(
    new THREE.Box3(
      new THREE.Vector3(mirroredMinX, bounds.min.y, bounds.min.z),
      new THREE.Vector3(mirroredMaxX, bounds.max.y, bounds.max.z),
    ),
  );
  const t = Math.max(0.0, Math.min(1.0, blendFactor));
  return new THREE.Box3(
    new THREE.Vector3(
      THREE.MathUtils.lerp(rightBounds.min.x, bothBounds.min.x, t),
      THREE.MathUtils.lerp(rightBounds.min.y, bothBounds.min.y, t),
      THREE.MathUtils.lerp(rightBounds.min.z, bothBounds.min.z, t),
    ),
    new THREE.Vector3(
      THREE.MathUtils.lerp(rightBounds.max.x, bothBounds.max.x, t),
      THREE.MathUtils.lerp(rightBounds.max.y, bothBounds.max.y, t),
      THREE.MathUtils.lerp(rightBounds.max.z, bothBounds.max.z, t),
    ),
  );
}

function getMedialMirrorPlaneX({
  bounds,
  sourceHemisphere,
}: {
  bounds: THREE.Box3;
  sourceHemisphere: HemisphereSide;
}): number {
  return sourceHemisphere === "left" ? bounds.max.x : bounds.min.x;
}

function cubicEaseInOut(x: number): number {
  const t = Math.max(0.0, Math.min(1.0, x));
  if (t < 0.5) {
    return 4.0 * t * t * t;
  }
  return 1.0 - Math.pow(-2.0 * t + 2.0, 3.0) / 2.0;
}

function stagedBlend(progress: number, start: number, end: number): number {
  if (end <= start) {
    throw new Error(`Invalid staged blend interval [${start}, ${end}].`);
  }
  return cubicEaseInOut((progress - start) / (end - start));
}

function wrapDegSigned(deg: number): number {
  return ((((deg + 180.0) % 360.0) + 360.0) % 360.0) - 180.0;
}

function formatSignedDegFixed(deg: number): string {
  const wrapped = wrapDegSigned(deg);
  const sign = wrapped < 0 ? "-" : "+";
  const absText = Math.abs(wrapped).toFixed(1).padStart(5, " ");
  return `${sign}${absText}`;
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
  elem.textContent =
    `camera yaw(Y)=${formatSignedDegFixed(azimuthY)}° ` +
    `pitch(X)=${formatSignedDegFixed(elevX)}° ` +
    `roll(Z)=${formatSignedDegFixed(rollZ)}° ` +
    `dist=${dist.toFixed(1)}`;
}

function updateCameraFollowLight({
  camera,
  controls,
  light,
}: {
  camera: THREE.PerspectiveCamera;
  controls: TrackballControls;
  light: THREE.DirectionalLight;
}): void {
  const target = controls.target.clone();
  const toCamera = camera.position.clone().sub(target);
  const distance = Math.max(toCamera.length(), 1.0);
  const elevated = toCamera
    .add(
      camera.up
        .clone()
        .normalize()
        .multiplyScalar(
          distance * Math.tan((CAMERA_LIGHT_ELEVATION_DEG * Math.PI) / 180.0),
        ),
    )
    .normalize()
    .multiplyScalar(distance);
  light.position.copy(target).add(elevated);
  light.target.position.copy(target);
  light.target.updateMatrixWorld();
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
  const bothHemispheresInput =
    getElement<HTMLInputElement>("both-hemispheres");
  const progressInput = getElement<HTMLInputElement>("progress");
  const playButton = getElement<HTMLButtonElement>("play");
  const speedSelect = getElement<HTMLSelectElement>("speed");
  const cameraReadoutElem = getElement<HTMLDivElement>("camera-readout");
  const scaleBarElem = getElement<HTMLDivElement>("scalebar");
  const scaleBarLabelElem = getElement<HTMLDivElement>("scalebar-label");

  const assetsBase = parseAssetsBase();
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

  const materialParams = {
    vertexColors: true,
    color: 0xffffff,
    side: THREE.DoubleSide,
  } as const satisfies THREE.MeshStandardMaterialParameters;
  const rightMaterial = new THREE.MeshStandardMaterial(materialParams);
  const leftMaterial = new THREE.MeshStandardMaterial(materialParams);
  leftMaterial.transparent = true;
  leftMaterial.opacity = 0.0;
  leftMaterial.depthWrite = false;
  const mesh = new THREE.Mesh(geometry, rightMaterial);
  const initialBounds = new THREE.Box3().setFromBufferAttribute(
    geometry.getAttribute("position") as THREE.BufferAttribute,
  );
  const sourceHemisphere = inferSourceHemisphere(loaded.positions, 5700.0);
  const mirroredMesh = new THREE.Mesh(geometry, leftMaterial);
  mirroredMesh.scale.x = -1.0;
  mirroredMesh.visible = false;
  const hemisphereSign = sourceHemisphere === "right" ? 1.0 : -1.0;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#f4f7ec");
  scene.add(mesh);
  scene.add(mirroredMesh);
  const labels = {
    anterior: createDirectionSprite("Anterior (-Z)", "#93354e"),
    posterior: createDirectionSprite("Posterior (+Z)", "#93354e"),
    negativeX: createDirectionSprite("Lateral (-X)", "#245f8f"),
    positiveX: createDirectionSprite("Medial (+X)", "#245f8f"),
  };
  scene.add(labels.anterior.sprite);
  scene.add(labels.posterior.sprite);
  scene.add(labels.negativeX.sprite);
  scene.add(labels.positiveX.sprite);
  const mlApPlaneGeometry = new THREE.PlaneGeometry(1, 1);
  const mlApPlane = new THREE.Mesh(
    mlApPlaneGeometry,
    new THREE.MeshBasicMaterial({
      color: 0x1f3b5c,
      transparent: true,
      opacity: PLANE_FILL_OPACITY,
      side: THREE.FrontSide,
      depthTest: true,
      depthWrite: true,
    }),
  );
  mlApPlane.rotation.x = -Math.PI / 2.0;
  mlApPlane.renderOrder = 2;
  scene.add(mlApPlane);
  const mlApPlaneOutline = new THREE.LineSegments(
    new THREE.EdgesGeometry(mlApPlaneGeometry),
    new THREE.LineBasicMaterial({
      color: 0x1f3b5c,
      transparent: true,
      opacity: PLANE_OUTLINE_OPACITY,
      depthTest: true,
      depthWrite: false,
    }),
  );
  mlApPlaneOutline.rotation.x = -Math.PI / 2.0;
  mlApPlaneOutline.renderOrder = 3;
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
  apAxisLine.renderOrder = 3;
  scene.add(apAxisLine);

  const ambient = new THREE.AmbientLight(0xffffff, 0.32);
  scene.add(ambient);
  const hemi = new THREE.HemisphereLight(0xfdfcf4, 0xc9d7b4, 1.15);
  scene.add(hemi);
  const key = new THREE.DirectionalLight(0xffffff, 0.75);
  key.position.set(2.0, 1.4, 1.1);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xd7e4ff, 0.32);
  rim.position.set(-1.4, 0.7, -1.3);
  scene.add(rim);
  const cameraLight = new THREE.DirectionalLight(
    0xfff7ef,
    CAMERA_LIGHT_INTENSITY,
  );
  scene.add(cameraLight);
  scene.add(cameraLight.target);

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
  let hemisphereBlend = bothHemispheresInput.checked ? 1.0 : 0.0;
  let hemisphereBlendTarget = hemisphereBlend;
  let activeBounds = computeActiveBounds({
    bounds: initialBounds,
    rightTranslateX: 0.0,
    mirroredPositionX: 0.0,
    blendFactor: hemisphereBlend,
  });
  let activeCenter = activeBounds.getCenter(new THREE.Vector3());
  let displayedMlLabelsShowBoth: boolean | null = null;
  const updateMlLabels = (showBoth: boolean): void => {
    if (displayedMlLabelsShowBoth === showBoth) {
      return;
    }
    displayedMlLabelsShowBoth = showBoth;
    const mlLabels = getMlAxisLabels({ sourceHemisphere, showBoth });
    setDirectionSpriteText(labels.negativeX, mlLabels.negativeX, "#245f8f");
    setDirectionSpriteText(labels.positiveX, mlLabels.positiveX, "#245f8f");
  };
  const syncHemisphereMode = (): void => {
    hemisphereBlendTarget = bothHemispheresInput.checked ? 1.0 : 0.0;
  };
  const updateHemispherePresentation = (bounds: THREE.Box3): void => {
    const overlayBlend = stagedBlend(
      hemisphereBlend,
      OVERLAY_STAGE_START_FRAC,
      OVERLAY_STAGE_END_FRAC,
    );
    const brainBlend = stagedBlend(hemisphereBlend, BRAIN_STAGE_START_FRAC, 1.0);
    updateMlLabels(overlayBlend >= 0.5);
    const mirrorPlaneX = getMedialMirrorPlaneX({ bounds, sourceHemisphere });
    const width = Math.max(bounds.max.x - bounds.min.x, 1.0);
    const rightTranslateX =
      hemisphereSign * HEMISPHERE_RIGHT_SLIDE_FRAC * width * brainBlend;
    const leftEntryOffset =
      hemisphereSign *
      HEMISPHERE_LEFT_ENTRY_FRAC *
      width *
      (1.0 - brainBlend);
    mesh.position.x = rightTranslateX;
    mirroredMesh.position.x = (2.0 * mirrorPlaneX) + leftEntryOffset;
    mirroredMesh.visible = brainBlend > 1.0e-3 || hemisphereBlendTarget >= 1.0;
    leftMaterial.opacity = brainBlend;
    leftMaterial.depthWrite = brainBlend >= 0.999;
    activeBounds = computeActiveBounds({
      bounds,
      rightTranslateX,
      mirroredPositionX: mirroredMesh.position.x,
      blendFactor: overlayBlend,
    });
    activeCenter = activeBounds.getCenter(new THREE.Vector3());
    (
      mlApPlane.material as THREE.MeshBasicMaterial
    ).opacity = PLANE_FILL_OPACITY;
    (
      mlApPlaneOutline.material as THREE.LineBasicMaterial
    ).opacity = PLANE_OUTLINE_OPACITY;
    (
      apAxisLine.material as THREE.LineBasicMaterial
    ).opacity = AP_AXIS_OPACITY;
  };

  const controls = new TrackballControls(camera, renderer.domElement);
  controls.rotateSpeed = 4.0;
  controls.panSpeed = 0.9;
  controls.zoomSpeed = 1.2;
  controls.dynamicDampingFactor = 0.1;
  controls.staticMoving = false;
  setCameraPose({
    camera,
    controls,
    target: activeCenter,
    yawDeg: DEFAULT_CAMERA_YAW_DEG,
    pitchDeg: DEFAULT_CAMERA_PITCH_DEG,
    rollDeg: DEFAULT_CAMERA_ROLL_DEG,
    dist: DEFAULT_CAMERA_DIST,
  });
  updateCameraFollowLight({ camera, controls, light: cameraLight });
  updateMlLabels(hemisphereBlendTarget > 0.5);
  syncHemisphereMode();

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
    applyUnfold(context, {
      progress,
      phase1Frac,
      flipXZ: false,
      outXYZ: dynamicPositions,
    });
    const positionAttr = geometry.getAttribute(
      "position",
    ) as THREE.BufferAttribute;
    positionAttr.needsUpdate = true;
    geometry.computeVertexNormals();
    const currentBounds = new THREE.Box3().setFromBufferAttribute(positionAttr);
    updateHemispherePresentation(currentBounds);
    updateDirectionSprites({ bounds: activeBounds, labels });
    updateApAxisLine({ bounds: activeBounds, line: apAxisLine });
    updateMlApPlane({
      bounds: activeBounds,
      planeFill: mlApPlane,
      planeOutline: mlApPlaneOutline,
    });
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
  bothHemispheresInput.addEventListener("change", () => {
    setPlaying(false);
    syncHemisphereMode();
  });
  playButton.addEventListener("click", () => setPlaying(!playing));

  applyCurrentState();

  const tick = (): void => {
    const nowSec = performance.now() * 0.001;
    const dt = Math.max(0.0, nowSec - lastTimeSec);
    lastTimeSec = nowSec;

    if (playing) {
      const speed = Number(speedSelect.value);
      progress = Math.min(1.0, progress + (dt * speed) / PLAYBACK_DURATION_SEC);
      progressInput.value = progress.toFixed(3);
      if (progress >= 1.0) {
        setPlaying(false);
      }
      applyCurrentState();
    }

    if (Math.abs(hemisphereBlend - hemisphereBlendTarget) > 1.0e-4) {
      const step = dt / HEMISPHERE_TRANSITION_SEC;
      if (hemisphereBlend < hemisphereBlendTarget) {
        hemisphereBlend = Math.min(hemisphereBlendTarget, hemisphereBlend + step);
      } else {
        hemisphereBlend = Math.max(hemisphereBlendTarget, hemisphereBlend - step);
      }
      applyCurrentState();
      if (Math.abs(hemisphereBlend - hemisphereBlendTarget) <= 1.0e-4) {
        hemisphereBlend = hemisphereBlendTarget;
      }
    }

    controls.update();
    updateCameraFollowLight({ camera, controls, light: cameraLight });
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
