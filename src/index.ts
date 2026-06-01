/**
 * vibe-plugin-aider
 *
 * Aider AI agent provider for VibeControls Agent.
 * CLI-only provider — spawns the CLI binary for prompt execution.
 */

import { Elysia } from "elysia";
import type {
  HostServices,
  VibePlugin,
  ProfileContext,
} from "@vibecontrols/plugin-sdk";
import {
  BoundLogger,
  ProviderRegistry,
  TelemetryEmitter,
  createLifecycleHooks,
} from "@vibecontrols/plugin-sdk";

// ── AI Provider Contract Types ──────────────────────────────────────────
// (provider-specific contract — kept inline; not part of the SDK surface)

type ProviderMode = "sdk" | "cli";

type AISessionStatus =
  | "active"
  | "idle"
  | "processing"
  | "error"
  | "terminated";
type AILogType =
  | "input"
  | "output"
  | "thinking"
  | "event"
  | "error"
  | "metadata";

export type PermissionMode = "plan" | "acceptEdits" | "fullAuto";

interface AISessionConfig {
  name: string;
  agentType: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
  systemPrompt?: string;
  workingDirectory?: string;
  /** Autonomy level for CLI mode. */
  permissionMode?: PermissionMode;
  providerConfig?: Record<string, unknown>;
}

interface AISession {
  id: string;
  name: string;
  status: AISessionStatus;
  agentType: string;
  provider: string;
  config: AISessionConfig;
  stats: AIUsageStats;
  createdAt: string;
  updatedAt: string;
}

interface AIContext {
  id: string;
  type: string;
  content: string;
  metadata?: Record<string, unknown>;
}
interface AIResponse {
  content: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
  durationMs: number;
  metadata?: Record<string, unknown>;
}
interface AIStreamChunk {
  type: "text" | "thinking" | "error" | "done";
  content: string;
  tokensUsed?: number;
}
interface AILog {
  id: string;
  sessionId: string;
  type: AILogType;
  content: string;
  tokenCount?: number;
  model?: string;
  durationMs?: number;
  createdAt: string;
}
interface AILogFilter {
  types?: AILogType[];
  startDate?: string;
  endDate?: string;
  search?: string;
  limit?: number;
  offset?: number;
}
interface AIUsageStats {
  inputTokens: number;
  outputTokens: number;
  requestCount: number;
  estimatedCostUsd: number;
}

interface AIAgentProvider {
  readonly name: string;
  createSession(config: AISessionConfig): Promise<AISession>;
  sendPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
  ): Promise<AIResponse>;
  streamPrompt?(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
    onChunk?: (chunk: AIStreamChunk) => void,
  ): Promise<AIResponse>;
  getSessionLogs(sessionId: string, filter?: AILogFilter): Promise<AILog[]>;
  getUsageStats(sessionId: string): Promise<AIUsageStats>;
  configureSession(
    sessionId: string,
    config: Partial<AISessionConfig>,
  ): Promise<void>;
  destroySession(sessionId: string): Promise<void>;
  listSessions(): Promise<AISession[]>;
  getSessionStatus(sessionId: string): Promise<AISessionStatus>;
  healthCheck(): Promise<{ ok: boolean; message?: string }>;
  getCapabilities?(): Record<string, boolean>;
  getMode?(): ProviderMode;
  setMode?(_mode: ProviderMode): void;
}

interface LogIngester {
  append(input: {
    sessionId: string;
    type: AILogType;
    content: string;
    tokenCount?: number;
    model?: string;
    durationMs?: number;
  }): unknown;
}

// ── Provider Implementation ──────────────────────────────────────────────

const PROVIDER_NAME = "aider";
const CLI_COMMAND = "aider";
/**
 * Resolve CLI binary path with platform-correct extension.
 * On Windows, Bun.spawn calls CreateProcess directly (no PATHEXT), so a bare
 * name won't find `name.exe`/`name.cmd`. Bun.which searches PATH like the shell.
 */
function platformExeName(base: string): string {
  return process.platform === "win32" ? `${base}.exe` : base;
}

function resolveCliBin(): string {
  const found =
    typeof Bun !== "undefined" && typeof Bun.which === "function"
      ? Bun.which(CLI_COMMAND, { PATH: process.env.PATH })
      : null;
  if (found) return found;
  return platformExeName(CLI_COMMAND);
}
const CLI_BIN = resolveCliBin();

const DISPLAY = "Aider";
const API_PREFIX = `/api/ai-${PROVIDER_NAME}`;
const SUPPORTED_MODES: ProviderMode[] = ["cli"];
// pip install args (without the interpreter). The Python executable is
// resolved dynamically at install time so Windows ("python") works as well as
// POSIX ("python3").
const CLI_INSTALL_ARGS = ["-m", "pip", "install", "--user", "aider-chat"];
const CLI_INSTALL_KIND = "pip" as const;

/**
 * Resolve the Python interpreter for the pip install spawn. Prefers `python3`
 * (canonical on POSIX) and falls back to `python` (canonical on Windows).
 * Returns null when neither is on PATH.
 */
function resolvePythonBin(): string | null {
  if (typeof Bun === "undefined" || typeof Bun.which !== "function")
    return null;
  // Pass the live PATH — Bun.which snapshots PATH at process start and would
  // otherwise miss a python installed/added to PATH mid-run.
  const opts = { PATH: process.env.PATH };
  return Bun.which("python3", opts) ?? Bun.which("python", opts);
}

interface ManagedSession {
  id: string;
  config: AISessionConfig;
  status: AISessionStatus;
  stats: AIUsageStats;
  createdAt: string;
  updatedAt: string;
}

class ProviderImpl implements AIAgentProvider {
  readonly name = PROVIDER_NAME;
  private sessions = new Map<string, ManagedSession>();
  private logIngester: LogIngester | null = null;
  private hostServices: HostServices | null = null;
  private logger: BoundLogger | null = null;

  setHostServices(hs: HostServices) {
    this.hostServices = hs;
    this.logger = new BoundLogger(hs.logger, `${PROVIDER_NAME}-provider`);
    const registry = new ProviderRegistry(hs);
    this.logIngester =
      registry.getProvider<LogIngester>("ai", "log-ingester") ?? null;
  }

  async createSession(config: AISessionConfig): Promise<AISession> {
    const id =
      (config.providerConfig?.["sessionId"] as string) || crypto.randomUUID();
    const now = new Date().toISOString();
    const existing = this.sessions.get(id);
    if (existing) {
      existing.status = "active";
      existing.updatedAt = now;
      return {
        id,
        name: existing.config.name,
        status: "active",
        agentType: existing.config.agentType,
        provider: PROVIDER_NAME,
        config: existing.config,
        stats: existing.stats,
        createdAt: existing.createdAt,
        updatedAt: now,
      };
    }
    const session: ManagedSession = {
      id,
      config,
      status: "active",
      stats: {
        inputTokens: 0,
        outputTokens: 0,
        requestCount: 0,
        estimatedCostUsd: 0,
      },
      createdAt: now,
      updatedAt: now,
    };
    this.sessions.set(id, session);
    return {
      id,
      name: config.name,
      status: "active",
      agentType: config.agentType,
      provider: PROVIDER_NAME,
      config,
      stats: session.stats,
      createdAt: now,
      updatedAt: now,
    };
  }

  async sendPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
  ): Promise<AIResponse> {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);
    if (session.status === "terminated")
      throw new Error("Session is terminated");
    session.status = "processing";
    session.updatedAt = new Date().toISOString();
    const startTime = Date.now();

    let fullPrompt = prompt;
    if (context?.length) {
      fullPrompt =
        prompt +
        "\n\n" +
        context
          .map((c) => `--- Context (${c.type}): ---\n${c.content}`)
          .join("\n\n");
    }

    this.logIngester?.append({ sessionId, type: "input", content: prompt });

    try {
      const args = CLI_ARGS_FN(session.config, fullPrompt);
      const proc = Bun.spawn([CLI_BIN, ...args], {
        stdout: "pipe",
        stderr: "pipe",
        cwd: session.config.workingDirectory || process.cwd(),
        timeout: 300_000,
      });
      const stdout = await new Response(proc.stdout).text();
      const stderr = await new Response(proc.stderr).text();
      const exitCode = await proc.exited;
      const durationMs = Date.now() - startTime;

      if (exitCode !== 0 && !stdout)
        throw new Error(`${DISPLAY} exited with code ${exitCode}: ${stderr}`);
      const content = stdout.trim() || stderr.trim();
      const inputTokens = Math.ceil(fullPrompt.length / 4);
      const outputTokens = Math.ceil(content.length / 4);
      session.stats.inputTokens += inputTokens;
      session.stats.outputTokens += outputTokens;
      session.stats.requestCount += 1;
      session.status = "active";
      session.updatedAt = new Date().toISOString();
      this.logIngester?.append({
        sessionId,
        type: "output",
        content,
        tokenCount: outputTokens,
        model: "default",
        durationMs,
      });
      return {
        content,
        model: "default",
        inputTokens,
        outputTokens,
        durationMs,
        metadata: { exitCode, provider: PROVIDER_NAME, mode: "cli" },
      };
    } catch (err) {
      session.status = "error";
      session.updatedAt = new Date().toISOString();
      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      this.logIngester?.append({ sessionId, type: "error", content: errorMsg });
      throw err;
    }
  }

  async getSessionLogs(
    _sessionId: string,
    _filter?: AILogFilter,
  ): Promise<AILog[]> {
    return [];
  }
  async getUsageStats(sessionId: string): Promise<AIUsageStats> {
    return (
      this.sessions.get(sessionId)?.stats ?? {
        inputTokens: 0,
        outputTokens: 0,
        requestCount: 0,
        estimatedCostUsd: 0,
      }
    );
  }
  async configureSession(
    sessionId: string,
    config: Partial<AISessionConfig>,
  ): Promise<void> {
    const s = this.sessions.get(sessionId);
    if (s) {
      Object.assign(s.config, config);
      s.updatedAt = new Date().toISOString();
    }
  }
  async destroySession(sessionId: string): Promise<void> {
    const s = this.sessions.get(sessionId);
    if (s) {
      s.status = "terminated";
      s.updatedAt = new Date().toISOString();
    }
  }
  async listSessions(): Promise<AISession[]> {
    return Array.from(this.sessions.values()).map((s) => ({
      id: s.id,
      name: s.config.name,
      status: s.status,
      agentType: s.config.agentType,
      provider: PROVIDER_NAME,
      config: s.config,
      stats: s.stats,
      createdAt: s.createdAt,
      updatedAt: s.updatedAt,
    }));
  }
  async getSessionStatus(sessionId: string): Promise<AISessionStatus> {
    return this.sessions.get(sessionId)?.status ?? "terminated";
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    try {
      const proc = Bun.spawnSync([CLI_BIN, "--version"], {
        timeout: 5000,
        stdout: "pipe",
        stderr: "ignore",
      });
      if (proc.exitCode === 0)
        return {
          ok: true,
          message: `${DISPLAY} ${proc.stdout.toString().trim()}`,
        };
      return { ok: false, message: `${DISPLAY} not available` };
    } catch {
      return { ok: false, message: `${DISPLAY} not installed` };
    }
  }

  getCapabilities() {
    return {
      streaming: false,
      vision: false,
      fileAttachments: false,
      toolUse: false,
      mcpSupport: false,
      voiceMode: false,
      cancelSupport: true,
      modelListing: false,
    };
  }
  getSupportedModes(): ProviderMode[] {
    return [...SUPPORTED_MODES];
  }
  getDisplayName(): string {
    return DISPLAY;
  }
  getPrereqApiPrefix(): string {
    return API_PREFIX;
  }
  getMode(): ProviderMode {
    return "cli";
  }
  setMode(mode: ProviderMode) {
    if (mode !== "cli") throw new Error(`${DISPLAY} only supports CLI mode`);
  }

  private log(level: "info" | "error" | "debug", msg: string) {
    this.logger?.[level](msg);
  }
}

// ── CLI Args Builder ─────────────────────────────────────────────────────

/**
 * Map the provider-agnostic permission mode to Aider flags.
 * Aider autonomy is coarse: `--yes-always` (auto-confirm everything) vs the
 * interactive default. `plan` uses architect/ask mode (`--chat-mode ask`) so
 * nothing is applied. Unknown/undefined → safe default (acceptEdits, which
 * lets aider apply edits but not blanket-confirm shell). CLI-version-dependent.
 */
export function permissionFlags(mode: PermissionMode | undefined): string[] {
  switch (mode) {
    case "plan":
      return ["--chat-mode", "ask"];
    case "fullAuto":
      return ["--yes-always"];
    case "acceptEdits":
    default:
      return [];
  }
}

function CLI_ARGS_FN(config: AISessionConfig, prompt: string): string[] {
  return ["--message", prompt, ...permissionFlags(config.permissionMode)];
}

// ── Plugin Export ────────────────────────────────────────────────────────

function getCliVersion(): string | null {
  try {
    const proc = Bun.spawnSync([CLI_BIN, "--version"], {
      timeout: 5000,
      stdout: "pipe",
      stderr: "ignore",
    });
    if (proc.exitCode === 0) return proc.stdout.toString().trim();
  } catch {
    // Binary not found.
  }
  return null;
}

function createPrereqsRoutes() {
  return new Elysia({ prefix: "/prereqs" })
    .get("/status", () => {
      const version = getCliVersion();
      return {
        satisfied: Boolean(version),
        missing: version
          ? []
          : [
              {
                name: CLI_COMMAND,
                kind: CLI_INSTALL_KIND,
                requiresSudo: false,
                description: `${DISPLAY} CLI for CLI mode`,
              },
            ],
      };
    })
    .post("/install", () => {
      if (getCliVersion())
        return {
          ok: true,
          installed: [CLI_COMMAND],
          pendingSudo: [],
          errors: [],
        };
      const py = resolvePythonBin();
      if (!py)
        return {
          ok: false,
          installed: [],
          pendingSudo: [],
          errors: [
            {
              name: CLI_COMMAND,
              message:
                "Python not found on PATH. Install Python 3, then run manually: " +
                `python3 ${CLI_INSTALL_ARGS.join(" ")}`,
            },
          ],
        };
      const proc = Bun.spawnSync([py, ...CLI_INSTALL_ARGS], {
        timeout: 120_000,
        stdout: "pipe",
        stderr: "pipe",
      });
      if (proc.exitCode === 0)
        return {
          ok: true,
          installed: [CLI_COMMAND],
          pendingSudo: [],
          errors: [],
        };
      return {
        ok: false,
        installed: [],
        pendingSudo: [],
        errors: [
          {
            name: CLI_COMMAND,
            message:
              proc.stderr.toString().trim() ||
              `Run manually: ${py} ${CLI_INSTALL_ARGS.join(" ")}`,
          },
        ],
      };
    });
}

const PLUGIN_NAME = PROVIDER_NAME;
const PLUGIN_VERSION = "1.0.0";

const provider = new ProviderImpl();

const lifecycle = createLifecycleHooks({
  name: PLUGIN_NAME,
  telemetryEventName: "ai.provider.ready",
  onInit: (hostServices: HostServices) => {
    provider.setHostServices(hostServices);
    new TelemetryEmitter(PLUGIN_NAME, PLUGIN_VERSION, hostServices).emit(
      "ai.provider.ready",
      { provider: PLUGIN_NAME },
    );
  },
  onShutdown: () => {
    for (const [id] of (provider as ProviderImpl)["sessions"]) {
      provider.destroySession(id).catch(() => {});
    }
  },
});

type AiderVibePlugin = VibePlugin & {
  providers?: { ai?: AIAgentProvider };
};

export const createPlugin = (_ctx: ProfileContext): AiderVibePlugin => ({
  capabilities: {
    secrets: "read",
    subprocess: true,
    gateway: false,
    telemetry: true,
  },
  name: PLUGIN_NAME,
  version: PLUGIN_VERSION,
  description: `${DISPLAY} AI agent provider for VibeControls`,
  tags: ["provider", "integration"],
  apiPrefix: API_PREFIX,
  prerequisites: [
    {
      name: CLI_COMMAND,
      kind: CLI_INSTALL_KIND,
      requiresSudo: false,
    },
  ],
  providers: { ai: provider },
  createRoutes: () => createPrereqsRoutes(),
  onServerStart: lifecycle.onServerStart,
  onServerStop: lifecycle.onServerStop,
});
