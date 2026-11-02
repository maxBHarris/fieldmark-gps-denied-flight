import { IMU_GRADES, NOISE_OPTIONS, type LatLon, type SimulateResponse } from '../api'
import { estimateRegionHalfWidthKm } from '../lib/geo'

interface ControlPanelProps {
  waypoints: LatLon[]
  planningMode: boolean
  onTogglePlanning: () => void
  onClearWaypoints: () => void
  onUndoWaypoint: () => void

  imuGrade: string
  onImuGradeChange: (v: string) => void
  magNoise: number
  onMagNoiseChange: (v: number) => void
  gravNoise: number
  onGravNoiseChange: (v: number) => void
  useMag: boolean
  onUseMagChange: (v: boolean) => void
  useGrav: boolean
  onUseGravChange: (v: boolean) => void
  speedMps: number
  onSpeedChange: (v: number) => void
  turnRateDps: number
  onTurnRateChange: (v: number) => void

  onRunSimulation: () => void
  loading: boolean
  error: string | null
  simResult: SimulateResponse | null

  currentIndex: number
  onCurrentIndexChange: (v: number) => void
  playing: boolean
  onTogglePlaying: () => void
  playbackSpeed: number
  onPlaybackSpeedChange: (v: number) => void
  sigmaMultiplier: number
  onSigmaMultiplierChange: (v: number) => void
}

const GRADE_LABELS: Record<string, string> = {
  consumer_mems: 'Consumer MEMS',
  tactical: 'Tactical',
  navigation: 'Navigation',
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2 border-b border-slate-800 pb-4">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</h3>
      {children}
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex items-center justify-between gap-3 text-sm text-slate-300">
      <span>{label}</span>
      {children}
    </label>
  )
}

const selectClass =
  'rounded-md border border-slate-700 bg-slate-800 px-2 py-1 text-sm text-slate-100 focus:border-sky-500 focus:outline-none'

export default function ControlPanel(props: ControlPanelProps) {
  const {
    waypoints, planningMode, onTogglePlanning, onClearWaypoints, onUndoWaypoint,
    imuGrade, onImuGradeChange, magNoise, onMagNoiseChange, gravNoise, onGravNoiseChange,
    useMag, onUseMagChange, useGrav, onUseGravChange, speedMps, onSpeedChange,
    turnRateDps, onTurnRateChange, onRunSimulation, loading, error, simResult,
    currentIndex, onCurrentIndexChange, playing, onTogglePlaying, playbackSpeed,
    onPlaybackSpeedChange, sigmaMultiplier, onSigmaMultiplierChange,
  } = props

  const maxIndex = simResult ? simResult.t.length - 1 : 0
  const regionHalfWidthKm = estimateRegionHalfWidthKm(waypoints, speedMps, turnRateDps)

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-4 text-slate-100">
      <Section title="Flight path">
        <button
          onClick={onTogglePlanning}
          className={`w-full rounded-md px-3 py-2 text-sm font-medium transition ${
            planningMode
              ? 'bg-sky-500 text-white'
              : 'bg-slate-800 text-slate-200 hover:bg-slate-700'
          }`}
        >
          {planningMode ? 'Click the globe to add waypoints…' : 'Plan a path'}
        </button>
        <div className="flex gap-2">
          <button
            onClick={onUndoWaypoint}
            disabled={waypoints.length === 0}
            className="flex-1 rounded-md bg-slate-800 px-2 py-1 text-xs text-slate-300 hover:bg-slate-700 disabled:opacity-40"
          >
            Undo last
          </button>
          <button
            onClick={onClearWaypoints}
            disabled={waypoints.length === 0}
            className="flex-1 rounded-md bg-slate-800 px-2 py-1 text-xs text-slate-300 hover:bg-slate-700 disabled:opacity-40"
          >
            Clear
          </button>
        </div>
        <p className="text-xs text-slate-500">{waypoints.length} waypoint(s) placed</p>
        {waypoints.length > 0 && (
          <p className={`text-xs ${regionHalfWidthKm > 600 ? 'text-amber-400' : 'text-slate-500'}`}>
            Map region: ~{Math.round(regionHalfWidthKm)} km half-width
            {regionHalfWidthKm > 600 ? ' (exceeds the 600 km limit -- shorten the route)' : ''}
          </p>
        )}
      </Section>

      <Section title="IMU">
        <Field label="Grade">
          <select
            className={selectClass}
            value={imuGrade}
            onChange={(e) => onImuGradeChange(e.target.value)}
          >
            {IMU_GRADES.map((g) => (
              <option key={g} value={g}>
                {GRADE_LABELS[g]}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Speed (m/s)">
          <input
            type="number"
            className={`${selectClass} w-20`}
            value={speedMps}
            min={20}
            max={400}
            onChange={(e) => onSpeedChange(Number(e.target.value))}
          />
        </Field>
        <Field label="Turn rate (deg/s)">
          <input
            type="number"
            className={`${selectClass} w-20`}
            value={turnRateDps}
            min={0.5}
            max={10}
            step={0.5}
            onChange={(e) => onTurnRateChange(Number(e.target.value))}
          />
        </Field>
      </Section>

      <Section title="Sensor noise (1-sigma)">
        <Field label="Magnetometer (nT)">
          <select
            className={selectClass}
            value={magNoise}
            onChange={(e) => onMagNoiseChange(Number(e.target.value))}
          >
            {NOISE_OPTIONS.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Gravimeter (mGal)">
          <select
            className={selectClass}
            value={gravNoise}
            onChange={(e) => onGravNoiseChange(Number(e.target.value))}
          >
            {NOISE_OPTIONS.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </Field>
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input type="checkbox" checked={useMag} onChange={(e) => onUseMagChange(e.target.checked)} />
          Use magnetic aiding
        </label>
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input type="checkbox" checked={useGrav} onChange={(e) => onUseGravChange(e.target.checked)} />
          Use gravity aiding
        </label>
      </Section>

      <button
        onClick={onRunSimulation}
        disabled={waypoints.length < 2 || loading}
        className="w-full rounded-md bg-emerald-600 px-3 py-2 text-sm font-semibold text-white transition hover:bg-emerald-500 disabled:opacity-40"
      >
        {loading ? 'Running…' : 'Run simulation'}
      </button>
      {error && <p className="text-xs text-red-400">{error}</p>}

      {simResult && (
        <Section title="Result">
          <div className="grid grid-cols-3 gap-2 text-center text-xs text-slate-300">
            <div>
              <div className="text-slate-500">Steady-state</div>
              <div className="font-mono text-sm">{simResult.tail_rms_m.toFixed(0)} m</div>
            </div>
            <div>
              <div className="text-slate-500">Max error</div>
              <div className="font-mono text-sm">{simResult.max_error_m.toFixed(0)} m</div>
            </div>
            <div>
              <div className="text-slate-500">Fixes</div>
              <div className="font-mono text-sm">
                {simResult.mag_updates}m/{simResult.grav_updates}g
              </div>
            </div>
          </div>

          <Field label="Corridor confidence">
            <select
              className={selectClass}
              value={sigmaMultiplier}
              onChange={(e) => onSigmaMultiplierChange(Number(e.target.value))}
            >
              <option value={1}>1σ (~68%)</option>
              <option value={2}>2σ (~95%)</option>
              <option value={3}>3σ (~99.7%)</option>
            </select>
          </Field>
          <p className="text-xs text-slate-500">
            The shaded corridor on the globe always shows the full route. Use the controls below
            to additionally inspect the true vs. estimated position at one moment.
          </p>

          <div className="flex items-center gap-2">
            <button
              onClick={onTogglePlaying}
              className="rounded-md bg-sky-600 px-3 py-1 text-sm font-medium text-white hover:bg-sky-500"
            >
              {playing ? 'Pause' : 'Play'}
            </button>
            <select
              className={selectClass}
              value={playbackSpeed}
              onChange={(e) => onPlaybackSpeedChange(Number(e.target.value))}
            >
              {[1, 2, 5, 10, 25].map((v) => (
                <option key={v} value={v}>
                  {v}x
                </option>
              ))}
            </select>
          </div>
          <input
            type="range"
            min={0}
            max={maxIndex}
            value={currentIndex}
            onChange={(e) => onCurrentIndexChange(Number(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-slate-400">
            <span>t = {simResult.t[currentIndex].toFixed(0)} s</span>
            <span>
              error {simResult.pos_error_m[currentIndex].toFixed(0)} m / 1σ{' '}
              {simResult.pos_std_m[currentIndex].toFixed(0)} m
            </span>
          </div>
        </Section>
      )}
    </div>
  )
}
