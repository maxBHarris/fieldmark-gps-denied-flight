import { useEffect, useRef, useState } from 'react'
import GlobeView from './components/Globe'
import ControlPanel from './components/ControlPanel'
import FieldMapsPanel from './components/FieldMapsPanel'
import SweepPanel from './components/SweepPanel'
import { api, type LatLon, type SimulateResponse } from './api'

type Tab = 'map' | 'sweep'

function App() {
  const [waypoints, setWaypoints] = useState<LatLon[]>([])
  const [planningMode, setPlanningMode] = useState(true)

  const [imuGrade, setImuGrade] = useState('tactical')
  const [magNoise, setMagNoise] = useState(1)
  const [gravNoise, setGravNoise] = useState(2)
  const [useMag, setUseMag] = useState(true)
  const [useGrav, setUseGrav] = useState(true)
  const [speedMps, setSpeedMps] = useState(180)
  const [turnRateDps, setTurnRateDps] = useState(3)

  const [simResult, setSimResult] = useState<SimulateResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [currentIndex, setCurrentIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(5)
  const [sigmaMultiplier, setSigmaMultiplier] = useState(2)

  const [tab, setTab] = useState<Tab>('map')

  async function runSimulation() {
    setLoading(true)
    setError(null)
    setPlaying(false)
    try {
      const result = await api.simulate({
        waypoints,
        imu_grade: imuGrade,
        mag_noise_nt: magNoise,
        grav_noise_mgal: gravNoise,
        use_mag: useMag,
        use_grav: useGrav,
        speed_mps: speedMps,
        turn_rate_dps: turnRateDps,
        dt: 1.0,
        seed: 0,
        n_grid: 81,
      })
      setSimResult(result)
      setCurrentIndex(0)
      setPlanningMode(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  // Playback loop: advance currentIndex while playing, stop at the end.
  const playingRef = useRef(playing)
  playingRef.current = playing
  useEffect(() => {
    if (!playing || !simResult) return
    const id = setInterval(() => {
      setCurrentIndex((i) => {
        const next = i + playbackSpeed
        if (next >= simResult.t.length - 1) {
          setPlaying(false)
          return simResult.t.length - 1
        }
        return next
      })
    }, 60)
    return () => clearInterval(id)
  }, [playing, playbackSpeed, simResult])

  function addWaypoint(coords: LatLon) {
    setWaypoints((w) => [...w, coords])
  }

  return (
    <div className="flex h-screen w-screen flex-col bg-slate-950">
      <header className="flex items-center justify-between border-b border-slate-800 px-4 py-2">
        <div>
          <h1 className="text-lg font-semibold text-slate-100">fieldmark</h1>
          <p className="text-xs text-slate-500">
            GPS-denied navigation via magnetic + gravity map matching (IGRF-14 + EGM2008)
          </p>
        </div>
        <nav className="flex gap-1 rounded-lg bg-slate-900 p-1">
          {(['map', 'sweep'] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`rounded-md px-3 py-1 text-sm font-medium transition ${
                tab === t ? 'bg-sky-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {t === 'map' ? 'Field maps' : 'Monte Carlo sweeps'}
            </button>
          ))}
        </nav>
      </header>

      <div className="flex min-h-0 flex-1">
        <aside className="w-80 flex-none border-r border-slate-800 bg-slate-900/50">
          <ControlPanel
            waypoints={waypoints}
            planningMode={planningMode}
            onTogglePlanning={() => setPlanningMode((p) => !p)}
            onClearWaypoints={() => {
              setWaypoints([])
              setSimResult(null)
            }}
            onUndoWaypoint={() => setWaypoints((w) => w.slice(0, -1))}
            imuGrade={imuGrade}
            onImuGradeChange={setImuGrade}
            magNoise={magNoise}
            onMagNoiseChange={setMagNoise}
            gravNoise={gravNoise}
            onGravNoiseChange={setGravNoise}
            useMag={useMag}
            onUseMagChange={setUseMag}
            useGrav={useGrav}
            onUseGravChange={setUseGrav}
            speedMps={speedMps}
            onSpeedChange={setSpeedMps}
            turnRateDps={turnRateDps}
            onTurnRateChange={setTurnRateDps}
            onRunSimulation={runSimulation}
            loading={loading}
            error={error}
            simResult={simResult}
            currentIndex={currentIndex}
            onCurrentIndexChange={(v) => {
              setPlaying(false)
              setCurrentIndex(v)
            }}
            playing={playing}
            onTogglePlaying={() => setPlaying((p) => !p)}
            playbackSpeed={playbackSpeed}
            onPlaybackSpeedChange={setPlaybackSpeed}
            sigmaMultiplier={sigmaMultiplier}
            onSigmaMultiplierChange={setSigmaMultiplier}
          />
        </aside>

        <main className="flex min-h-0 flex-1 flex-col">
          <div className="min-h-0 flex-1 p-2">
            <GlobeView
              waypoints={waypoints}
              planningMode={planningMode}
              onAddWaypoint={addWaypoint}
              simResult={simResult}
              currentIndex={currentIndex}
              sigmaMultiplier={sigmaMultiplier}
            />
          </div>
          <div className="h-[380px] flex-none border-t border-slate-800 bg-slate-900/30">
            {tab === 'map' ? (
              <FieldMapsPanel region={simResult?.region ?? null} />
            ) : (
              <SweepPanel
                waypoints={waypoints}
                imuGrade={imuGrade}
                speedMps={speedMps}
                turnRateDps={turnRateDps}
              />
            )}
          </div>
        </main>
      </div>
    </div>
  )
}

export default App
