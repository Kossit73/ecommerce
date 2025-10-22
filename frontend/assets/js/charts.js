import { toast } from './dom.js';

export function renderPlotlyChart(targetId, traces, layoutOverrides = {}) {
  const node = document.getElementById(targetId);
  if (!node) return;
  if (!traces || traces.length === 0) {
    node.innerHTML = '<p class="status">No chart data available.</p>';
    return;
  }
  const layout = {
    margin: { t: 40, r: 20, l: 50, b: 40 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    hovermode: 'x unified',
    font: { family: 'Inter, sans-serif' },
    ...layoutOverrides,
  };
  if (window.Plotly && window.Plotly.react) {
    window.Plotly.react(node, traces, layout, { responsive: true });
  } else {
    toast('Plotly failed to load. Charts are unavailable.', { tone: 'error' });
  }
}
