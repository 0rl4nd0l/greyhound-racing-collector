# MCP in this project

This folder provides a project-scoped Model Context Protocol setup for Claude Desktop and for local testing via the MCP Inspector.

What’s included
- claude_desktop_config.example.json: Example Claude Desktop config wiring up:
  - filesystem server scoped to this repo path
  - fetch server for making HTTP requests
- This README with quick commands.

Prereqs
- macOS, zsh
- Node.js 18+ (check with: node -v)

Use with Claude Desktop
1) Open (or create) your user config:
   - Path: ~/Library/Application Support/Claude/claude_desktop_config.json
2) Merge the example from this directory into your user config. Minimal working config:

{
  "mcpServers": {
    "filesystem_greyhound": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "--allowed-dirs",
        "/Users/test/Desktop/greyhound_racing_collector"
      ]
    },
    "fetch": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-fetch"
      ]
    }
  }
}

3) Save and restart Claude Desktop.

Test locally with MCP Inspector
- Install the inspector: npm i -g @modelcontextprotocol/inspector
- Launch a test session exposing this repo:

mcp-inspector --server 'npx -y @modelcontextprotocol/server-filesystem --allowed-dirs /Users/test/Desktop/greyhound_racing_collector' --server 'npx -y @modelcontextprotocol/server-fetch'

Security notes
- Filesystem server is scoped to this repo only. Add more paths with additional --allowed-dirs arguments if needed.
- Avoid running a shell MCP server unless strictly necessary.

Project hygiene
- Keep all MCP-related files under tools/mcp/.
- If you later add custom MCP servers, place them under tools/mcp/custom-*/ and move any deprecated scripts to archive/ as per repo rules.

