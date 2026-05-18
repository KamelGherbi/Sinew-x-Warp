import { spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { execFile as execFileCallback } from "node:child_process";

const execFile = promisify(execFileCallback);
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, "..");
const tauriBin = path.join(
  projectRoot,
  "node_modules",
  ".bin",
  process.platform === "win32" ? "tauri.cmd" : "tauri",
);
const args = process.argv.slice(2);

if (shouldUseMacosDmgWorkaround(args)) {
  await buildMacosAppThenDmg(args);
} else {
  process.exit(await run(tauriBin, args));
}

function shouldUseMacosDmgWorkaround(args) {
  if (process.platform !== "darwin") return false;
  if (args[0] !== "build") return false;
  if (args.includes("--no-bundle")) return false;
  const bundles = bundleTargets(args);
  return bundles === null || bundles.includes("dmg") || bundles.includes("all");
}

async function buildMacosAppThenDmg(args) {
  const passthrough = stripBundleArgs(args.slice(1));
  const config = process.env.TAURI_SIGNING_PRIVATE_KEY
    ? []
    : ["--config", '{"bundle":{"createUpdaterArtifacts":false}}'];
  const status = await run(tauriBin, [
    "build",
    "--bundles",
    "app",
    ...config,
    ...passthrough,
  ]);
  if (status !== 0) process.exit(status);
  await createMacosDmg();
}

function bundleTargets(args) {
  const index = args.findIndex((arg) => arg === "--bundles" || arg === "-b");
  if (index === -1) return null;
  const raw = args[index + 1];
  if (!raw || raw.startsWith("-")) return [];
  return raw.split(/[ ,]+/).filter(Boolean);
}

function stripBundleArgs(args) {
  const next = [];
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === "--bundles" || arg === "-b") {
      i += 1;
      continue;
    }
    next.push(arg);
  }
  return next;
}

async function createMacosDmg() {
  const releaseDir = await macosReleaseDir();
  const bundleDir = path.join(releaseDir, "bundle");
  const macosDir = path.join(bundleDir, "macos");
  const dmgDir = path.join(bundleDir, "dmg");
  const appPath = path.join(macosDir, "Sinew.app");
  const iconPath = path.join(projectRoot, "src-tauri", "icons", "icon.icns");
  const packageJson = JSON.parse(
    await fs.readFile(path.join(projectRoot, "package.json"), "utf8"),
  );
  const arch = process.arch === "arm64" ? "aarch64" : "x64";
  const dmgName = `Sinew_${packageJson.version}_${arch}.dmg`;
  const dmgPath = path.join(dmgDir, dmgName);
  const staging = await fs.mkdtemp(path.join(os.tmpdir(), "sinew-dmg-"));
  const stagedApp = path.join(staging, "Sinew.app");

  await fs.mkdir(dmgDir, { recursive: true });
  await fs.rm(dmgPath, { force: true });
  await detachStaleSinewDmgs(macosDir, dmgDir);
  await fs.rm(path.join(macosDir, `rw.${process.pid}.${dmgName}`), { force: true });
  await execFile("ditto", ["--noextattr", "--noqtn", appPath, stagedApp]);
  await execFile("ln", ["-s", "/Applications", path.join(staging, "Applications")]);
  await execFile("hdiutil", [
    "create",
    "-srcfolder",
    staging,
    "-volname",
    "Sinew Installer",
    "-fs",
    "HFS+",
    "-format",
    "UDZO",
    "-imagekey",
    "zlib-level=9",
    dmgPath,
  ]);
  await fs.rm(staging, { recursive: true, force: true });
  await execFile("hdiutil", ["internet-enable", "-yes", dmgPath]).catch(() => undefined);
  console.log(`       Built DMG at: ${dmgPath}`);
}

async function macosReleaseDir() {
  const universal = path.join(projectRoot, "target", "universal-apple-darwin", "release");
  try {
    await fs.access(path.join(universal, "bundle", "macos"));
    return universal;
  } catch {
    return path.join(projectRoot, "target", "release");
  }
}

async function detachStaleSinewDmgs(...dirs) {
  let plist;
  try {
    ({ stdout: plist } = await execFile("hdiutil", ["info", "-plist"]));
  } catch {
    return;
  }
  const plistPath = path.join(os.tmpdir(), `sinew-hdiutil-${process.pid}.plist`);
  try {
    await fs.writeFile(plistPath, plist);
    const { stdout } = await execFile("plutil", ["-convert", "json", "-o", "-", plistPath]);
    const data = JSON.parse(stdout);
    for (const image of Array.isArray(data.images) ? data.images : []) {
      const imagePath = String(image["image-path"] ?? "");
      if (!dirs.some((dir) => imagePath.startsWith(dir))) continue;
      const entities = Array.isArray(image["system-entities"])
        ? image["system-entities"]
        : [];
      const device = entities
        .map((entity) => String(entity["dev-entry"] ?? ""))
        .find((entry) => /^\/dev\/disk\d+$/.test(entry)) ??
        entities
          .map((entity) => String(entity["dev-entry"] ?? ""))
          .find((entry) => entry.startsWith("/dev/disk"));
      if (!device) continue;
      await execFile("hdiutil", ["detach", device]).catch(() =>
        execFile("hdiutil", ["detach", "-force", device]).catch(() => undefined),
      );
    }
  } finally {
    await fs.rm(plistPath, { force: true }).catch(() => undefined);
  }
}

function run(command, args) {
  return new Promise((resolve) => {
    const child = spawn(command, args, {
      cwd: projectRoot,
      env: process.env,
      stdio: "inherit",
      shell: false,
    });
    child.on("close", (code) => resolve(code ?? 1));
    child.on("error", (err) => {
      console.error(err);
      resolve(1);
    });
  });
}
