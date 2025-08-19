import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

// Minimal MCP server exposing tools relevant to this repo.
// - get_recent_runs: returns a short message about where to find recent race/test results.
// - get_repo_paths: returns structured key locations to help the client browse context safely.
// - list_recent_archived_races: lists recent archived race CSVs (archive-first policy).
// - locate_missing_files: searches for filenames, prioritizing archive folders.

function getRepoRoot(): string {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  // dist -> custom-greyhound-server -> mcp -> tools -> repo root
  return path.resolve(__dirname, "../../../..");
}

async function main() {
  const server = new McpServer({
    name: "greyhound-custom-mcp",
    version: "0.2.0"
  });

  // get_recent_runs
  server.registerTool(
    "get_recent_runs",
    {
      title: "Get Recent Runs",
      description: "Summarize where to find recent test and race outputs in this repo.",
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: "text",
          text: [
            "Key places to inspect for recent work:",
            "- REAL_RACE_TEST_RESULTS.md",
            "- SMOKE_TEST_RESULTS.md",
            "- .github/workflows/* for CI runs",
            "- archive/ for any archived or legacy outputs.",
            "Note: winner data must be scraped from the race page itself (not the form guide)."
          ].join("\n")
        }
      ]
    })
  );

  // get_repo_paths
  server.registerTool(
    "get_repo_paths",
    {
      title: "Get Repo Paths",
      description: "Return important repo paths (docs, scripts, archives) to guide safe browsing.",
      inputSchema: {}
    },
    async () => ({
      content: [
        {
          type: "text",
          text: JSON.stringify({
            docs: [
              "README.md",
              "PROJECT_CATALOGUE.md",
              "DATABASE_SCHEMA_DOCUMENTATION.md"
            ],
            mcp: [
              "tools/mcp/README.md",
              "tools/mcp/custom-greyhound-server"
            ],
            archives: [
              "archive/",
              "archive/corrupt_or_legacy_race_files/"
            ],
            scripts: [
              "app.py",
              "advanced_system_analyzer.py",
              "advanced_ensemble_ml_system.py"
            ]
          }, null, 2)
        }
      ]
    })
  );

  // list_recent_archived_races
  server.registerTool(
    "list_recent_archived_races",
    {
      title: "List Recent Archived Races",
      description: "List the most recent archived race CSV files (archive-first policy).",
      inputSchema: {
        limit: z.number().optional(),
        subdir: z.string().optional()
      }
    },
    async ({ limit, subdir }: { limit?: number; subdir?: string }) => {
      const lim = Math.max(1, Math.min(100, Number(limit ?? 10)));
      const sub = String(subdir ?? "archive/corrupt_or_legacy_race_files");
      const repoRoot = getRepoRoot();
      const targetDir = path.resolve(repoRoot, sub);

      let entries: { name: string; full: string; mtime: number }[] = [];
      try {
        const files = await fs.promises.readdir(targetDir);
        for (const f of files) {
          if (!f.toLowerCase().endsWith(".csv")) continue;
          const full = path.join(targetDir, f);
          try {
            const stat = await fs.promises.stat(full);
            entries.push({ name: f, full, mtime: stat.mtimeMs });
          } catch {}
        }
      } catch (e: any) {
        return { content: [{ type: "text", text: `Failed to read ${sub}: ${e?.message ?? e}` }], isError: true } as any;
      }

      entries.sort((a, b) => b.mtime - a.mtime);
      const top = entries.slice(0, lim).map(e => ({ file: e.name, mtime_iso: new Date(e.mtime).toISOString() }));

      return {
        content: [{ type: "text", text: JSON.stringify({ subdir: sub, count: top.length, files: top }, null, 2) }]
      };
    }
  );

  // locate_missing_files
  server.registerTool(
    "locate_missing_files",
    {
      title: "Locate Missing Files",
      description: "Search for filenames across the repo, prioritizing archive directories before suggesting new files.",
      inputSchema: {
        names: z.array(z.string()),
        search_dirs: z.array(z.string()).optional()
      }
    },
    async ({ names, search_dirs }: { names: string[]; search_dirs?: string[] }) => {
      const repoRoot = getRepoRoot();
      const searchDirs = Array.isArray(search_dirs) && search_dirs.length > 0 ? search_dirs : ["archive", "."];

      const MAX_FILES_SCANNED = 5000;
      let scanned = 0;

      const results: Record<string, string[]> = Object.fromEntries(names.map((n: string) => [path.basename(n), [] as string[]]));

      async function walk(dir: string) {
        if (scanned >= MAX_FILES_SCANNED) return;
        let items: fs.Dirent[] = [];
        try {
          items = await fs.promises.readdir(dir, { withFileTypes: true });
        } catch { return; }
        for (const it of items) {
          if (scanned >= MAX_FILES_SCANNED) break;
          const full = path.join(dir, it.name);
          if (it.isDirectory()) {
            if (["node_modules", ".git", "dist", ".venv", "__pycache__"].includes(it.name)) continue;
            await walk(full);
          } else if (it.isFile()) {
            scanned++;
            const base = it.name;
            if (results[base] !== undefined) {
              results[base].push(path.relative(repoRoot, full));
            }
          }
        }
      }

      for (const rel of searchDirs) {
        const start = path.resolve(repoRoot, rel);
        await walk(start);
        if (names.every((n: string) => results[path.basename(n)].length > 0)) break;
      }

      return {
        content: [{ type: "text", text: JSON.stringify({ searched: searchDirs, results, scanned }, null, 2) }]
      };
    }
  );

  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

