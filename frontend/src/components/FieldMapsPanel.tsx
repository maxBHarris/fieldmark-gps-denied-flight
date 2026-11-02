import Plotly from 'plotly.js-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'
import type { Data, Layout } from 'plotly.js'
import type { RegionData } from '../api'

const Plot = createPlotlyComponent(Plotly)

const baseLayout: Partial<Layout> = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: { color: '#cbd5e1', size: 11 },
  margin: { l: 50, r: 10, t: 36, b: 40 },
  xaxis: { title: { text: 'East (km)' } },
  yaxis: { title: { text: 'North (km)' }, scaleanchor: 'x' },
}

export default function FieldMapsPanel({ region }: { region: RegionData | null }) {
  if (!region) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-slate-500">
        Run a simulation to see the magnetic and gravity reference maps for its region.
      </div>
    )
  }

  const magTrace: Partial<Data> = {
    type: 'heatmap',
    x: region.east_km,
    y: region.north_km,
    z: region.mag_grid,
    colorscale: 'Viridis',
    colorbar: { title: { text: 'nT' } },
  }
  const gravTrace: Partial<Data> = {
    type: 'heatmap',
    x: region.east_km,
    y: region.north_km,
    z: region.grav_grid,
    colorscale: 'RdBu',
    reversescale: true,
    colorbar: { title: { text: 'mGal' } },
  }

  return (
    <div className="grid h-full grid-cols-1 gap-4 overflow-y-auto p-4 md:grid-cols-2">
      <div>
        <Plot
          data={[magTrace]}
          layout={{ ...baseLayout, title: { text: 'Magnetic total intensity' }, height: 420 }}
          style={{ width: '100%' }}
          useResizeHandler
          config={{ displayModeBar: false }}
        />
      </div>
      <div>
        <Plot
          data={[gravTrace]}
          layout={{ ...baseLayout, title: { text: 'Gravity disturbance' }, height: 420 }}
          style={{ width: '100%' }}
          useResizeHandler
          config={{ displayModeBar: false }}
        />
      </div>
      <p className="text-xs text-slate-500 md:col-span-2">
        Centered at {region.lat0.toFixed(3)}, {region.lon0.toFixed(3)}, ±{region.half_width_km.toFixed(0)} km.
        Magnetic = real IGRF-14 core field + synthetic crustal texture. Gravity = real EGM2008
        (degree 360), referenced to the WGS84 normal ellipsoid.
      </p>
    </div>
  )
}
