import * as THREE from "three";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls.js";

import { buildColors, type ColorMode } from "./color";
import { loadIjkAssets } from "./io";

declare const __REPO_ROOT_ABS__: string;

const canvasRoot = document.getElementById("canvas-root") as HTMLDivElement;
const statusElem = document.getElementById("status") as HTMLDivElement;
const bboxElem = document.getElementById("bbox") as HTMLDivElement;
const errorElem = document.getElementById("error") as HTMLDivElement;
const pointSizeElem = document.getElementById("point-size") as HTMLInputElement;
const colorModeElem = document.getElementById("color-mode") as HTMLSelectElement;
const resetCameraElem = document.getElementById("reset-camera") as HTMLButtonElement;

const scene = new THREE.Scene();
scene.background = new THREE.Color("#eef2e8");

const camera = new THREE.PerspectiveCamera(
  55,
  canvasRoot.clientWidth / canvasRoot.clientHeight,
  0.1,
  1_000_000,
);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(canvasRoot.clientWidth, canvasRoot.clientHeight);
canvasRoot.appendChild(renderer.domElement);

const controls = new TrackballControls(camera, renderer.domElement);
controls.rotateSpeed = 4.0;
controls.panSpeed = 0.9;
controls.zoomSpeed = 1.2;
controls.dynamicDampingFactor = 0.1;
controls.staticMoving = false;
controls.handleResize();

scene.add(new THREE.HemisphereLight(0xfaf9ef, 0xb9c7a0, 0.85));
const keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
keyLight.position.set(2.0, 1.4, 1.1);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0xc8d7ff, 0.45);
rimLight.position.set(-1.4, 0.7, -1.3);
scene.add(rimLight);

type ColorStats = {
  r_um_min: number;
  r_um_max: number;
  t_lookup_min: number;
  t_lookup_max: number;
};

let points: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial> | null = null;
let pointsBounds: THREE.Box3 | null = null;
let colorStats: ColorStats | null = null;
let leidenNCategories: number | null = null;
let guidePlaneFill: THREE.Mesh | null = null;
let guidePlaneOutline: THREE.LineSegments | null = null;
let apAxisLine: THREE.Line | null = null;
type DirectionSprite = { sprite: THREE.Sprite; aspect: number };
type DirectionLabels = {
  anterior: DirectionSprite;
  posterior: DirectionSprite;
  lateral: DirectionSprite;
  medial: DirectionSprite;
};
let directionLabels: DirectionLabels | null = null;

function defaultAssetsDir(): string {
  return `/@fs${__REPO_ROOT_ABS__}/results/refextract/all_h5ad_threejs_ijk_assets`;
}

function setError(message: string): void {
  errorElem.textContent = message;
}

function clearError(): void {
  errorElem.textContent = "";
}

function createCirclePointTexture(): THREE.CanvasTexture {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (ctx === null) {
    throw new Error("Failed to create canvas context for point texture.");
  }
  const cx = size * 0.5;
  const cy = size * 0.5;
  const r = size * 0.42;

  ctx.clearRect(0, 0, size, size);
  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
  grad.addColorStop(0.0, "rgba(255,255,255,1.0)");
  grad.addColorStop(0.75, "rgba(255,255,255,1.0)");
  grad.addColorStop(1.0, "rgba(255,255,255,0.0)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, 2.0 * Math.PI);
  ctx.closePath();
  ctx.fill();

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

function createDirectionSprite(text: string, colorHex: string): DirectionSprite {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  if (context === null) {
    throw new Error("Failed to create 2D canvas context for direction label.");
  }

  const fontPx = 22;
  const padX = 12;
  const padY = 8;
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

function ensureDirectionLabels(): void {
  if (directionLabels !== null) {
    return;
  }
  directionLabels = {
    anterior: createDirectionSprite("Anterior (-Z)", "#93354e"),
    posterior: createDirectionSprite("Posterior (+Z)", "#93354e"),
    lateral: createDirectionSprite("Lateral (-X)", "#245f8f"),
    medial: createDirectionSprite("Medial (+X)", "#245f8f"),
  };
  scene.add(directionLabels.anterior.sprite);
  scene.add(directionLabels.posterior.sprite);
  scene.add(directionLabels.lateral.sprite);
  scene.add(directionLabels.medial.sprite);
}

function updateDirectionLabels(bounds: THREE.Box3): void {
  ensureDirectionLabels();
  if (directionLabels === null) {
    throw new Error("Direction labels were not initialized.");
  }
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.z, 1.0);
  const apPad = 0.45 * extent;
  const y = center.y;
  const textH = Math.max(7.0, 0.009 * Math.max(size.length(), 1.0));
  const mlPad = Math.max(0.26 * extent, 1.25 * textH);

  directionLabels.anterior.sprite.position.set(center.x, y, bounds.min.z - apPad);
  directionLabels.posterior.sprite.position.set(center.x, y, bounds.max.z + apPad);
  directionLabels.lateral.sprite.position.set(bounds.min.x - mlPad, y, center.z);
  directionLabels.medial.sprite.position.set(bounds.max.x + mlPad, y, center.z);

  directionLabels.anterior.sprite.scale.set(textH * directionLabels.anterior.aspect, textH, 1.0);
  directionLabels.posterior.sprite.scale.set(textH * directionLabels.posterior.aspect, textH, 1.0);
  directionLabels.lateral.sprite.scale.set(textH * directionLabels.lateral.aspect, textH, 1.0);
  directionLabels.medial.sprite.scale.set(textH * directionLabels.medial.aspect, textH, 1.0);
}

function ensureGuideObjects(): void {
  if (guidePlaneFill === null) {
    const fillGeom = new THREE.PlaneGeometry(1.0, 1.0);
    const fillMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(0x1f3b5c),
      transparent: true,
      opacity: 0.13,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
    });
    guidePlaneFill = new THREE.Mesh(fillGeom, fillMat);
    guidePlaneFill.rotation.x = -Math.PI * 0.5;
    guidePlaneFill.renderOrder = 1;
    scene.add(guidePlaneFill);
  }
  if (guidePlaneOutline === null) {
    const edgeGeom = new THREE.EdgesGeometry(new THREE.PlaneGeometry(1.0, 1.0));
    const edgeMat = new THREE.LineBasicMaterial({
      color: new THREE.Color(0x1f3b5c),
      transparent: true,
      opacity: 0.7,
      depthWrite: false,
      depthTest: true,
    });
    guidePlaneOutline = new THREE.LineSegments(edgeGeom, edgeMat);
    guidePlaneOutline.rotation.x = -Math.PI * 0.5;
    guidePlaneOutline.renderOrder = 2;
    scene.add(guidePlaneOutline);
  }
  if (apAxisLine === null) {
    const axisGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(),
      new THREE.Vector3(),
    ]);
    const axisMat = new THREE.LineBasicMaterial({
      color: new THREE.Color(0x8b1c1c),
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
      depthTest: true,
    });
    apAxisLine = new THREE.Line(axisGeom, axisMat);
    apAxisLine.renderOrder = 3;
    scene.add(apAxisLine);
  }
}

function updateGuideObjects(bounds: THREE.Box3): void {
  ensureGuideObjects();
  if (guidePlaneFill === null || guidePlaneOutline === null || apAxisLine === null) {
    throw new Error("Guide objects were not initialized.");
  }
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.z, 1.0);

  const padPlane = 0.15 * extent;
  const width = Math.max(1.0, size.x + 2.0 * padPlane);
  const depth = Math.max(1.0, size.z + 2.0 * padPlane);
  guidePlaneFill.position.set(center.x, center.y, center.z);
  guidePlaneFill.scale.set(width, depth, 1.0);
  guidePlaneOutline.position.set(center.x, center.y, center.z);
  guidePlaneOutline.scale.set(width, depth, 1.0);

  const padAxis = 0.5 * extent;
  const linePos = apAxisLine.geometry.getAttribute("position") as THREE.BufferAttribute;
  linePos.setXYZ(0, center.x, center.y, bounds.min.z - padAxis);
  linePos.setXYZ(1, center.x, center.y, bounds.max.z + padAxis);
  linePos.needsUpdate = true;
  apAxisLine.geometry.computeBoundingSphere();
  updateDirectionLabels(bounds);
}

function fitCameraToBounds(bounds: THREE.Box3): void {
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const radius = Math.max(size.length() * 0.5, 1.0);
  camera.position.set(center.x + radius * 1.2, center.y + radius * 0.7, center.z + radius * 1.4);
  camera.near = Math.max(radius * 0.001, 0.1);
  camera.far = Math.max(radius * 30.0, 10_000);
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

function refreshColor(mode: ColorMode): void {
  if (points === null || colorStats === null) {
    return;
  }
  const geom = points.geometry;
  const colorAttr = geom.getAttribute("color") as THREE.BufferAttribute;
  const rUmAttr = geom.getAttribute("r_um") as THREE.BufferAttribute;
  const tLookupAttr = geom.getAttribute("t_lookup") as THREE.BufferAttribute;
  const axisAttr = geom.getAttribute("axis_code") as THREE.BufferAttribute;
  const leidenAttr = geom.getAttribute("leiden_code") as THREE.BufferAttribute | undefined;

  const colors = colorAttr.array as Uint8Array;
  buildColors({
    mode,
    rUm: rUmAttr.array as Float32Array,
    tLookup: tLookupAttr.array as Float32Array,
    axis: axisAttr.array as Uint8Array,
    leiden: leidenAttr === undefined ? undefined : (leidenAttr.array as Uint16Array),
    leidenNCategories: leidenNCategories === null ? undefined : leidenNCategories,
    stats: colorStats,
    outRgb: colors,
  });
  colorAttr.needsUpdate = true;
}

async function init(): Promise<void> {
  try {
    clearError();
    const params = new URLSearchParams(window.location.search);
    const assetsDirRaw = params.get("assets");
    const assetsDir =
      assetsDirRaw !== null && assetsDirRaw.trim() !== ""
        ? assetsDirRaw.trim()
        : defaultAssetsDir();

    statusElem.textContent = `Loading assets from ${assetsDir} ...`;
    const loaded = await loadIjkAssets(assetsDir);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(loaded.positions, 3));

    const colors = new Uint8Array(loaded.manifest.n_points * 3);
    buildColors({
      mode: "r_um",
      rUm: loaded.rUm,
      tLookup: loaded.tLookup,
      axis: loaded.axis,
      stats: loaded.manifest.stats,
      outRgb: colors,
    });

    const colorAttr = new THREE.Uint8BufferAttribute(colors, 3, true);
    geometry.setAttribute("color", colorAttr);

    geometry.setAttribute("r_um", new THREE.BufferAttribute(loaded.rUm, 1));
    geometry.setAttribute("t_lookup", new THREE.BufferAttribute(loaded.tLookup, 1));

    geometry.setAttribute("axis_code", new THREE.Uint8BufferAttribute(loaded.axis, 1));
    if (loaded.leiden !== undefined) {
      geometry.setAttribute("leiden_code", new THREE.Uint16BufferAttribute(loaded.leiden, 1));
      leidenNCategories = loaded.leidenCategories?.length ?? null;
    } else {
      leidenNCategories = null;
    }
    colorStats = {
      r_um_min: loaded.manifest.stats.r_um_min,
      r_um_max: loaded.manifest.stats.r_um_max,
      t_lookup_min: loaded.manifest.stats.t_lookup_min,
      t_lookup_max: loaded.manifest.stats.t_lookup_max,
    };

    const material = new THREE.PointsMaterial({
      size: Number(pointSizeElem.value),
      sizeAttenuation: true,
      vertexColors: true,
      map: createCirclePointTexture(),
      alphaTest: 0.35,
      opacity: 0.9,
      transparent: true,
      depthWrite: false,
    });

    points = new THREE.Points(geometry, material);
    scene.add(points);

    geometry.computeBoundingBox();
    if (geometry.boundingBox === null) {
      throw new Error("Failed to compute point-cloud bounds.");
    }
    pointsBounds = geometry.boundingBox.clone();
    updateGuideObjects(pointsBounds);

    fitCameraToBounds(pointsBounds);

    const bmin = loaded.manifest.stats.bbox_xyz_min;
    const bmax = loaded.manifest.stats.bbox_xyz_max;
    const leidenText =
      loaded.leidenCategories === undefined
        ? ""
        : `, leiden=${loaded.leidenCategories.length.toLocaleString()}`;
    statusElem.textContent = `Loaded ${loaded.manifest.n_points.toLocaleString()} points (coronal=${loaded.manifest.axis_counts.coronal.toLocaleString()}, sagittal=${loaded.manifest.axis_counts.sagittal.toLocaleString()}${leidenText})`;
    bboxElem.textContent = `bbox xyz min=[${bmin.map((v) => v.toFixed(2)).join(", ")}], max=[${bmax.map((v) => v.toFixed(2)).join(", ")}]`;

    if (loaded.leidenCategories !== undefined) {
      const opt = colorModeElem.querySelector('option[value="leiden"]') as HTMLOptionElement | null;
      if (opt !== null) {
        opt.disabled = false;
      }
      colorModeElem.value = "leiden";
      refreshColor("leiden");
    } else {
      const opt = colorModeElem.querySelector('option[value="leiden"]') as HTMLOptionElement | null;
      if (opt !== null) {
        opt.disabled = true;
      }
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    setError(message);
    statusElem.textContent = "Load failed.";
  }
}

pointSizeElem.addEventListener("input", () => {
  if (points !== null) {
    points.material.size = Number(pointSizeElem.value);
    points.material.needsUpdate = true;
  }
});

colorModeElem.addEventListener("change", () => {
  const mode = colorModeElem.value as ColorMode;
  refreshColor(mode);
});

resetCameraElem.addEventListener("click", () => {
  if (pointsBounds !== null) {
    fitCameraToBounds(pointsBounds);
  }
});

window.addEventListener("resize", () => {
  const width = canvasRoot.clientWidth;
  const height = canvasRoot.clientHeight;
  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  controls.handleResize();
});

function animate(): void {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

void init();
animate();
