# Custom Greyhound MCP Server

This directory contains a minimal custom MCP server tailored for the greyhound_racing_collector repo. It supplements the filesystem and fetch servers already documented in tools/mcp/README.md.

What it does
- Exposes example tools:
  - get_recent_runs: Pointers to where recent results/logs live in this repo
  - get_repo_paths: Key paths (docs, scripts, archives, mcp) to guide safe browsing

Install and run (local)
1) From this directory:
   npm install
2) Dev mode (auto-reload):
   npm run dev
3) Build and run:
   npm run build && npm start

Use with MCP Inspector
mcp-inspector \
  --server "node dist/index.js" \
  --server 'npx -y @modelcontextprotocol/server-filesystem --allowed-dirs /Users/test/Desktop/greyhound_racing_collector' \
  --server 'npx -y @modelcontextprotocol/server-fetch'

Wire into Claude Desktop
Add this entry to your Claude config (merge with existing):
{
  "mcpServers": {
    "greyhound_custom": {
      "command": "node",
      "args": [
        "/Users/test/Desktop/greyhound_racing_collector/tools/mcp/custom-greyhound-server/dist/index.js"
      ]
    }
  }
}

Repo hygiene
- All MCP files live under tools/mcp/
- If you replace this server later, move the old one to archive/ in line with repo rules.

