import React from 'react';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'];

export default function PortfolioChart({ allocations }) {
  // allocations is the array of dicts we sent from Flask: 
  // [{ ticker: 'AAPL', value: 25000, weight: 0.25 }, ...]
  
  // If no data yet, show a placeholder
  if (!allocations || allocations.length === 0) {
    return <div className="h-64 flex items-center justify-center text-gray-400">No data to display</div>;
  }

  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 h-96">
      <h3 className="text-lg font-bold text-gray-800 mb-4">Optimal Asset Allocation</h3>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={allocations}
            // innerRadius creates the donut hole 
            innerRadius={80} 
            outerRadius={120}
            paddingAngle={5}
            dataKey="value"
            nameKey="ticker"
          >
            {allocations.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip 
            formatter={(value, name, props) => [
              `$${value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`, 
              `${name} (${(props.payload.weight * 100).toFixed(1)}%)`
            ]} 
          />
          <Legend verticalAlign="bottom" height={36}/>
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}