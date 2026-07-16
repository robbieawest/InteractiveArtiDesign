import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  // relative asset paths so the build works both locally and when served
  // from a subpath like https://<user>.github.io/InteractiveArtiDesign/
  base: "./",
});
