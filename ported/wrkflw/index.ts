import express from "express";
import cors from "cors";
import { DateTime } from "luxon";
import bodyParser from "body-parser";

const app = express();
app.use(cors());
app.use(bodyParser.json());

const DEFAULT_ZONES = [
  "UTC",
  "America/New_York",
  "America/Los_Angeles",
  "Europe/London",
  "Europe/Berlin",
  "Asia/Tokyo",
  "Australia/Sydney"
];

// Mock data stores (in production, use a real database)
let projects = [
  {
    id: "p1",
    name: "Mock Project A",
    location: "123 Main St",
    startDate: "2025-09-01",
    endDate: "2025-12-01",
    members: ["u1", "u2", "u3"]
  }
];

let dailyLogs = [
  {
    id: "dl1",
    projectId: "p1",
    authorId: "u1",
    date: "2025-09-08",
    notes: "Initial setup and site preparation completed.",
    attachments: []
  }
];

// Health check endpoint
app.get("/api/health", (_req, res) => res.json({ status: "ok", now: new Date().toISOString() }));

// Timezone endpoints
app.get("/api/timezones", (_req, res) => {
  res.json({ timezones: DEFAULT_ZONES });
});

app.post("/api/time", (req, res) => {
  const zones: string[] = Array.isArray(req.body?.timezones) ? req.body.timezones : DEFAULT_ZONES;
  try {
    const times = zones.map((zone) => {
      const dt = DateTime.now().setZone(zone);
      return {
        timezone: zone,
        iso: dt.toISO(),
        formatted: dt.toLocaleString(DateTime.DATETIME_FULL_WITH_SECONDS)
      };
    });
    res.json({ times });
  } catch (err) {
    res.status(400).json({ error: "Invalid timezone(s) provided", details: String(err) });
  }
});

// Projects endpoints
app.get("/api/projects", (_req, res) => {
  res.json({ projects });
});

app.post("/api/projects", (req, res) => {
  const { name, location, startDate, endDate, members } = req.body;
  if (!name) {
    return res.status(400).json({ error: "Project name is required" });
  }
  
  const project = {
    id: `p${Date.now()}`,
    name,
    location: location || "",
    startDate: startDate || new Date().toISOString().split('T')[0],
    endDate: endDate || "",
    members: members || []
  };
  
  projects.push(project);
  res.status(201).json({ project });
});

app.get("/api/projects/:projectId", (req, res) => {
  const project = projects.find(p => p.id === req.params.projectId);
  if (!project) {
    return res.status(404).json({ error: "Project not found" });
  }
  res.json({ project });
});

// Daily logs endpoints
app.get("/api/projects/:projectId/daily-logs", (req, res) => {
  const projectLogs = dailyLogs.filter(log => log.projectId === req.params.projectId);
  res.json({ dailyLogs: projectLogs });
});

app.post("/api/projects/:projectId/daily-logs", (req, res) => {
  const { projectId } = req.params;
  const { authorId, notes, attachments } = req.body;
  
  // Check if project exists
  const project = projects.find(p => p.id === projectId);
  if (!project) {
    return res.status(404).json({ error: "Project not found" });
  }
  
  const dailyLog = {
    id: `dl${Date.now()}`,
    projectId,
    authorId: authorId || "u1", // Default to u1 if not provided
    date: new Date().toISOString().split('T')[0],
    notes: notes || "",
    attachments: attachments || []
  };
  
  dailyLogs.push(dailyLog);
  res.status(201).json({ dailyLog });
});

const PORT = process.env.PORT ? parseInt(process.env.PORT) : 4000;
app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`API listening on http://localhost:${PORT}`);
});