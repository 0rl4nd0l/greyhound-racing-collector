import fs from "fs";
import path from "path";

// Create a minimal upcoming race CSV if none exist. Returns true if created.
export async function ensureUpcomingRaceCsv(): Promise<boolean> {
  try {
    const upcomingDir = path.resolve("upcoming_races");
    if (!fs.existsSync(upcomingDir)) {
      fs.mkdirSync(upcomingDir, { recursive: true });
    }

    // If already has CSVs, do nothing
    const existing = fs
      .readdirSync(upcomingDir)
      .filter((f) => f.endsWith(".csv") && !f.startsWith("."));
    if (existing.length) return false;

    const today = new Date();
    const y = today.getFullYear();
    const m = String(today.getMonth() + 1).padStart(2, "0");
    const d = String(today.getDate()).padStart(2, "0");

    const filename = `Race_1_WPK_${y}-${m}-${d}.csv`;
    const filePath = path.join(upcomingDir, filename);

    const content = [
      "Dog Name,BOX,WGT,TRAINER,SP,DIST,G,PERF,SPEED,CLASS",
      "1. Test Dog A,1,30.2,Trainer A,3.2,520,G5,0.5,0.6,0.4",
      "2. Test Dog B,2,29.8,Trainer B,4.5,520,G5,0.4,0.5,0.3",
      "3. Test Dog C,3,31.1,Trainer C,6.0,520,G5,0.3,0.4,0.2",
      "4. Test Dog D,4,30.7,Trainer D,8.0,520,G5,0.2,0.3,0.1",
    ].join("\n");

    fs.writeFileSync(filePath, content, { encoding: "utf-8" });
    return true;
  } catch (e) {
    console.error("Failed to ensure upcoming race CSV:", e);
    return false;
  }
}

