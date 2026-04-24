import { useEffect, useRef } from 'react'

function LossChart({ losses, currentIndex }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current

    if (!canvas) {
      return
    }

    const width = canvas.clientWidth || 260
    const height = 132
    const ctx = canvas.getContext('2d')
    const safeLosses = losses.length > 0 ? losses : [0]
    const clampedIndex = Math.min(currentIndex, safeLosses.length - 1)
    const chartLeft = 14
    const chartRight = width - 14
    const chartTop = 12
    const chartBottom = height - 30
    const chartWidth = Math.max(chartRight - chartLeft, 1)
    const chartHeight = Math.max(chartBottom - chartTop, 1)

    canvas.width = width
    canvas.height = height

    ctx.clearRect(0, 0, width, height)
    ctx.fillStyle = 'rgba(8, 18, 26, 0.82)'
    ctx.fillRect(0, 0, width, height)

    let maxLoss = Math.max(...safeLosses)
    let minLoss = Math.min(...safeLosses)
    let lossRange = maxLoss - minLoss

    if (lossRange < 0.001) {
      const midpoint = (maxLoss + minLoss) / 2
      minLoss = Math.max(0, midpoint - 0.0005)
      maxLoss = midpoint + 0.0005
      lossRange = maxLoss - minLoss
    } else {
      const padding = lossRange * 0.16
      minLoss = Math.max(0, minLoss - padding)
      maxLoss += padding
      lossRange = maxLoss - minLoss
    }

    for (let lineIndex = 0; lineIndex < 4; lineIndex += 1) {
      const y = chartTop + (chartHeight / 3) * lineIndex

      ctx.beginPath()
      ctx.moveTo(chartLeft, y)
      ctx.lineTo(chartRight, y)
      ctx.strokeStyle = 'rgba(112, 154, 182, 0.16)'
      ctx.lineWidth = 1
      ctx.stroke()
    }

    const getPoint = (loss, index) => {
      const x = chartLeft + (index / Math.max(safeLosses.length - 1, 1)) * chartWidth
      const normalizedLoss = (loss - minLoss) / lossRange
      const y = chartBottom - normalizedLoss * chartHeight

      return { x, y }
    }

    const fillGradient = ctx.createLinearGradient(0, chartTop, 0, chartBottom)
    fillGradient.addColorStop(0, 'rgba(142, 234, 255, 0.34)')
    fillGradient.addColorStop(1, 'rgba(142, 234, 255, 0.05)')

    const strokeGradient = ctx.createLinearGradient(chartLeft, chartTop, chartRight, chartBottom)
    strokeGradient.addColorStop(0, '#8ef0ff')
    strokeGradient.addColorStop(1, '#38cfe0')

    ctx.beginPath()

    safeLosses.forEach((loss, index) => {
      const { x, y } = getPoint(loss, index)

      if (index === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })

    ctx.lineTo(chartRight, chartBottom)
    ctx.lineTo(chartLeft, chartBottom)
    ctx.closePath()
    ctx.fillStyle = fillGradient
    ctx.fill()

    ctx.beginPath()

    safeLosses.forEach((loss, index) => {
      const { x, y } = getPoint(loss, index)

      if (index === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })

    ctx.strokeStyle = strokeGradient
    ctx.lineWidth = 2.5
    ctx.shadowColor = '#78deff'
    ctx.shadowBlur = 12
    ctx.stroke()
    ctx.shadowBlur = 0

    safeLosses.forEach((loss, index) => {
      const { x, y } = getPoint(loss, index)

      ctx.beginPath()
      ctx.arc(x, y, index === clampedIndex ? 4.5 : 3.2, 0, Math.PI * 2)
      ctx.fillStyle = index === clampedIndex ? '#ff9f43' : '#3ed2dd'
      ctx.fill()
    })

    const { x: markerX, y: markerY } = getPoint(safeLosses[clampedIndex], clampedIndex)

    ctx.beginPath()
    ctx.moveTo(markerX, chartTop)
    ctx.lineTo(markerX, chartBottom)
    ctx.strokeStyle = 'rgba(255, 211, 138, 0.45)'
    ctx.lineWidth = 1
    ctx.stroke()

    ctx.beginPath()
    ctx.arc(markerX, markerY, 5, 0, Math.PI * 2)
    ctx.fillStyle = '#ff9f43'
    ctx.fill()

    ctx.fillStyle = 'rgba(190, 217, 235, 0.92)'
    ctx.font = '600 11px IBM Plex Mono, monospace'
    ctx.textBaseline = 'alphabetic'
    ctx.fillText('Epoca 1', chartLeft, height - 8)

    const lastEpochLabel = `Epoca ${safeLosses.length}`
    const labelWidth = ctx.measureText(lastEpochLabel).width
    ctx.fillText(lastEpochLabel, chartRight - labelWidth, height - 8)
  }, [currentIndex, losses])

  return (
    <div id="loss-chart">
      <canvas id="loss-canvas" ref={canvasRef} />
    </div>
  )
}

export default LossChart
