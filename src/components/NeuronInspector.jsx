function NeuronInspector({ neuron, visible }) {
  return (
    <div id="info" className={visible ? 'on' : ''}>
      <h3>{neuron?.title ?? 'Neurona'}</h3>
      <div className="row">
        <span className="lbl">Capa</span>
        <span className="val">{neuron?.layerName ?? '—'}</span>
      </div>
      <div className="row">
        <span className="lbl">Indice</span>
        <span className="val">{neuron?.index ?? '—'}</span>
      </div>
      <div className="row">
        <span className="lbl">Activacion</span>
        <span className={`val ${neuron?.activationTone ?? ''}`}>
          {neuron?.activation ?? '—'}
        </span>
      </div>
      <div className="row">
        <span className="lbl">Funcion</span>
        <span className="val">{neuron?.functionName ?? '—'}</span>
      </div>

      {neuron?.isInputLayer ? (
        <div className="sec">Input layer</div>
      ) : (
        <>
          <div className="sec">Incoming weights</div>
          {neuron?.incomingWeights?.map((weight) => (
            <div className="wb" key={weight.label}>
              <div className="wbl">{weight.label}</div>
              <div className="wbt">
                <div
                  className="wbf"
                  style={{ width: `${weight.percent}%`, background: weight.color }}
                />
              </div>
            </div>
          ))}
        </>
      )}

      <div className="hint">clic en vacio · cerrar</div>
    </div>
  )
}

export default NeuronInspector
