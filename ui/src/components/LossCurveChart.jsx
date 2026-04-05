import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const COLORS = [
  "#60a5fa", // blue  – train / class_0
  "#f87171", // red   – val   / class_1
  "#4ade80", // green – class_2
  "#facc15", // yellow– class_3
  "#c084fc", // purple– class_4
  "#fb923c", // orange
];

export default function LossCurveChart({ curves, label, yLabel = "Loss" }) {
  if (!curves || Object.keys(curves).length === 0) return null;

  const keys = Object.keys(curves);
  const maxLen = Math.max(...keys.map((k) => (curves[k] || []).length));

  const data = Array.from({ length: maxLen }, (_, i) => {
    const point = { epoch: i + 1 };
    keys.forEach((k) => {
      const v = curves[k]?.[i];
      point[k] = v !== undefined && !isNaN(v) ? +v.toFixed(4) : null;
    });
    return point;
  });

  return (
    <div className="chart-wrapper">
      <p className="chart-label">{label}</p>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
          <XAxis
            dataKey="epoch"
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            label={{ value: "Epoch", position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 11 }}
          />
          <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} width={48} />
          <Tooltip
            contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6 }}
            labelStyle={{ color: "#94a3b8" }}
            itemStyle={{ color: "#e2e8f0" }}
          />
          <Legend wrapperStyle={{ fontSize: 12, color: "#94a3b8" }} />
          {keys.map((k, i) => (
            <Line
              key={k}
              type="monotone"
              dataKey={k}
              stroke={COLORS[i % COLORS.length]}
              dot={false}
              strokeWidth={2}
              connectNulls={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
