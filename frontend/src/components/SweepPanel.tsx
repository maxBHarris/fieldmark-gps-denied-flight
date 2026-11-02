import { useState } from 'react'
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api, type DriftResponse, type LatLon, type Sweep2DResponse, type SweepResponse } from '../api'

interface SweepPanelProps {
  waypoints: LatLon[]
  imuGrade: string
  speedMps: number
  turnRateDps: number
}

const GRADE_COLORS: Record<string, string> = {
  consumer_mems: '#f87171',
  tactical: '#facc15',
  navigation: '#4ade80',
}

function toChartData(sweep: SweepResponse) {
  return sweep.values.map((v, i) => ({ value: v, mean: sweep.means[i], std: sweep.stds[i] }))
}

function toDriftChartData(drift: DriftResponse) {
  return drift.horizons_s.map((h, i) => {
    const row: Record<string, number> = { horizon: h }
    for (const grade of Object.keys(drift.by_grade)) row[grade] = drift.by_grade[grade][i]
    return row
  })
}

function Heatmap2D({ sweep }: { sweep: Sweep2DResponse }) {
  const flat = sweep.grid.flat()
  const min = Math.min(...flat)
  const max = Math.max(...flat)
  const colorFor = (v: number) => {
    const t = max > min ? (v - min) / (max - min) : 0
    const hue = 140 - 140 * t // green (low error) -> red (high error)
    return `hsl(${hue}, 70%, 35%)`
  }
  return (
    <div className="inline-block">
      <div
        className="grid gap-px"
        style={{ gridTemplateColumns: `auto repeat(${sweep.mag_values.length}, 2.75rem)` }}
      >
        <div />
        {sweep.mag_values.map((m) => (
          <div key={m} className="text-center text-[10px] text-slate-400">
            {m}
          </div>
        ))}
        {sweep.grav_values.map((g, i) => (
          <>
            <div key={`label-${g}`} className="pr-1 text-right text-[10px] text-slate-400">
              {g}
            </div>
            {sweep.mag_values.map((_, j) => (
              <div
                key={`${i}-${j}`}
                className="flex h-9 w-11 items-center justify-center text-[10px] text-white"
                style={{ backgroundColor: colorFor(sweep.grid[i][j]) }}
                title={`${sweep.grid[i][j].toFixed(0)} m`}
              >
                {sweep.grid[i][j].toFixed(0)}
              </div>
            ))}
          </>
        ))}
      </div>
      <p className="mt-1 text-[10px] text-slate-500">
        rows = gravimeter noise (mGal), columns = magnetometer noise (nT); cell = median steady-state error (m)
      </p>
    </div>
  )
}

export default function SweepPanel({ waypoints, imuGrade, speedMps, turnRateDps }: SweepPanelProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [nTrials, setNTrials] = useState(5)
  const [magSweep, setMagSweep] = useState<SweepResponse | null>(null)
  const [gravSweep, setGravSweep] = useState<SweepResponse | null>(null)
  const [sweep2d, setSweep2d] = useState<Sweep2DResponse | null>(null)
  const [drift, setDrift] = useState<DriftResponse | null>(null)

  const canRun = waypoints.length >= 2 && !loading

  async function runAnalysis() {
    setLoading(true)
    setError(null)
    try {
      const sweepParams = {
        waypoints, imu_grade: imuGrade, speed_mps: speedMps, turn_rate_dps: turnRateDps,
        dt: 1.0, n_trials: nTrials, n_grid: 81,
      }
      const driftParams = { waypoints, speed_mps: speedMps, turn_rate_dps: turnRateDps, dt: 1.0, n_trials: nTrials }
      const [mag, grav, grid2d, driftRes] = await Promise.all([
        api.sweepMag(sweepParams),
        api.sweepGrav(sweepParams),
        api.sweep2d(sweepParams),
        api.drift(driftParams),
      ])
      setMagSweep(mag)
      setGravSweep(grav)
      setSweep2d(grid2d)
      setDrift(driftRes)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-4">
      <div className="flex items-center gap-3">
        <button
          onClick={runAnalysis}
          disabled={!canRun}
          className="rounded-md bg-emerald-600 px-3 py-2 text-sm font-semibold text-white hover:bg-emerald-500 disabled:opacity-40"
        >
          {loading ? 'Running Monte Carlo…' : 'Run Monte Carlo analysis'}
        </button>
        <label className="flex items-center gap-2 text-sm text-slate-300">
          Trials per point
          <input
            type="number"
            min={2}
            max={15}
            value={nTrials}
            onChange={(e) => setNTrials(Number(e.target.value))}
            className="w-16 rounded-md border border-slate-700 bg-slate-800 px-2 py-1 text-sm"
          />
        </label>
        {waypoints.length < 2 && (
          <span className="text-xs text-slate-500">Plan a path first (2+ waypoints)</span>
        )}
      </div>
      {error && <p className="text-xs text-red-400">{error}</p>}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {magSweep && (
          <div>
            <h4 className="mb-1 text-sm font-medium text-slate-300">
              Accuracy vs. magnetometer noise (magnetic-only aiding)
            </h4>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={toChartData(magSweep)}>
                <CartesianGrid stroke="#1e293b" />
                <XAxis dataKey="value" scale="log" type="number" domain={['auto', 'auto']}
                  stroke="#94a3b8" tick={{ fontSize: 11 }} label={{ value: 'nT', position: 'insideBottom', offset: -4, fill: '#64748b' }} />
                <YAxis scale="log" domain={['auto', 'auto']} stroke="#94a3b8" tick={{ fontSize: 11 }}
                  label={{ value: 'm', angle: -90, position: 'insideLeft', fill: '#64748b' }} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e293b' }} />
                <Line type="monotone" dataKey="mean" stroke="#38bdf8" dot strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
        {gravSweep && (
          <div>
            <h4 className="mb-1 text-sm font-medium text-slate-300">
              Accuracy vs. gravimeter noise (gravity-only aiding)
            </h4>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={toChartData(gravSweep)}>
                <CartesianGrid stroke="#1e293b" />
                <XAxis dataKey="value" scale="log" type="number" domain={['auto', 'auto']}
                  stroke="#94a3b8" tick={{ fontSize: 11 }} label={{ value: 'mGal', position: 'insideBottom', offset: -4, fill: '#64748b' }} />
                <YAxis scale="log" domain={['auto', 'auto']} stroke="#94a3b8" tick={{ fontSize: 11 }}
                  label={{ value: 'm', angle: -90, position: 'insideLeft', fill: '#64748b' }} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e293b' }} />
                <Line type="monotone" dataKey="mean" stroke="#f472b6" dot strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
        {drift && (
          <div>
            <h4 className="mb-1 text-sm font-medium text-slate-300">
              Unaided drift by IMU grade
            </h4>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={toDriftChartData(drift)}>
                <CartesianGrid stroke="#1e293b" />
                <XAxis dataKey="horizon" stroke="#94a3b8" tick={{ fontSize: 11 }}
                  label={{ value: 's', position: 'insideBottom', offset: -4, fill: '#64748b' }} />
                <YAxis scale="log" domain={['auto', 'auto']} stroke="#94a3b8" tick={{ fontSize: 11 }}
                  label={{ value: 'm', angle: -90, position: 'insideLeft', fill: '#64748b' }} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e293b' }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {Object.keys(drift.by_grade).map((grade) => (
                  <Line key={grade} type="monotone" dataKey={grade} stroke={GRADE_COLORS[grade]} dot strokeWidth={2} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
        {sweep2d && (
          <div>
            <h4 className="mb-1 text-sm font-medium text-slate-300">
              Joint sensitivity (matched update rates)
            </h4>
            <Heatmap2D sweep={sweep2d} />
          </div>
        )}
      </div>
    </div>
  )
}
