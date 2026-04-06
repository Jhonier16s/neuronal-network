function TopBar({ architectureLabel, connectionCount, epoch, mouseLabel }) {
  return (
    <div id="top-bar">
      <div id="ttl">
        <h1>Neural Network</h1>
        <p>{mouseLabel}</p>
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
