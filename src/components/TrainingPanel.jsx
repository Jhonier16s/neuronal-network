import LossChart from './LossChart.jsx'

function TrainingPanel({
  training,
  onStepBack,
  onStepForward,
  onStepEpochForward,
  onReset,
  cinematicActive,
}) {
  return (
    <div id="train-panel">
      <h3>Entrenamiento Excel</h3>

      <div className="train-row">
        <span className="train-lbl">Paso</span>
        <span className="train-val">{training.step}</span>
      </div>
      <div className="train-row">
        <span className="train-lbl">Loss</span>
        <span className={`train-val ${training.lossTone}`}>{training.lossText}</span>
      </div>
      <div className="train-row">
        <span className="train-lbl">LR</span>
        <span className="train-val">{training.lrText}</span>
      </div>
      <div className="train-row">
        <span className="train-lbl">Estado</span>
        <span className={`train-val ${training.statusTone}`}>{training.statusText}</span>
      </div>
      <div className="train-row train-row-stack">
        <span className="train-lbl">Muestra</span>
        <span className="train-val mono">{training.sampleText}</span>
      </div>
      <div className="train-row train-row-stack">
        <span className="train-lbl">Visual</span>
        <span className="train-val mono dimmed">{training.visualText}</span>
      </div>

      <div className="signal-legend" aria-hidden="true">
        <span className="signal-chip forward">Forward</span>
        <span className="signal-chip backward">Backprop</span>
        <span className="signal-chip field">Salida</span>
      </div>

      <LossChart losses={training.losses} currentIndex={training.currentLossIndex} />

      <div id="step-controls">
        <button className="step-btn" onClick={onStepBack} disabled={cinematicActive}>
          Atras
        </button>
        <button className="step-btn step-btn-reset" onClick={onReset} disabled={cinematicActive}>
          Reiniciar
        </button>
        <button className="step-btn primary" onClick={onStepForward} disabled={cinematicActive}>
          Siguiente muestra
        </button>
      </div>

      <button
        className="step-btn step-btn-epoch"
        onClick={onStepEpochForward}
        disabled={cinematicActive}
      >
        Siguiente epoca
      </button>

      <div id="step-pos">{training.stepPosition}</div>
    </div>
  )
}

export default TrainingPanel
