import { resolve } from "node:path";
import { defineConfig } from "vite";

const here = resolve(__dirname);
const repoRoot = resolve(here, "../../..");

export default defineConfig({
  define: {
    __VIEWER_DIR_ABS__: JSON.stringify(here),
    __REPO_ROOT_ABS__: JSON.stringify(repoRoot),
  },
  server: {
    fs: {
      allow: [here, resolve(here, ".."), resolve(here, "../.."), repoRoot],
    },
  },
});
