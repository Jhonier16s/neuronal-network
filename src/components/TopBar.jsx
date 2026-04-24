function TopBar({ architectureLabel, connectionCount, epoch, mouseLabel, hintActive }) {
  return (
    <div id="top-bar">
      <div id="ttl">
        <h1>AI Training Chamber</h1>
        <p className={hintActive ? 'hint-pulse' : ''}>
          {hintActive ? <span className="hint-dot" aria-hidden="true" /> : null}
          {mouseLabel}
        </p>
      </div>

      <div id="stats">
        <div>
          Arquitectura <b>{architectureLabel}</b>
        </div>
        <div>
          Pesos <b>{connectionCount}</b> · Epoca <b>{epoch}</b>
        </div>
      </div>
    </div>
  )
}

export default TopBar
