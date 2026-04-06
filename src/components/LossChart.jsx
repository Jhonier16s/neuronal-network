import { useEffect, useRef } from 'react'

function LossChart({ losses, currentIndex }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current

    if (!canvas) {
      return
    }

    const width = canvas.clientWidth || 260
    const height = 48
    const ctx = canvas.getContext('2d')

    canvas.width = width
    canvas.height = height

    ctx.clearRect(0, 0, width, height)

    if (losses.length < 2) {
      return
    }

    let maxLoss = Math.max(...losses)

    if (maxLoss < 0.001) {
      maxLoss = 0.001
    }

    ctx.strokeStyle = '#1a2a38'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, height / 2)
    ctx.lineTo(width, height / 2)
    ctx.stroke()

    ctx.beginPath()

    losses.forEach((loss, index) => {
      const x = (index / (losses.length - 1)) * width
      const y = height - (loss / maxLoss) * (height - 4) - 2

      if (index === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })

    ctx.strokeStyle = '#4a90d9'
    ctx.lineWidth = 1.5
    ctx.stroke()

    const markerX =
      (currentIndex / Math.max(losses.length - 1, 1)) * width

    ctx.beginPath()
    ctx.moveTo(markerX, 0)
    ctx.lineTo(markerX, height)
    ctx.strokeStyle = '#6ab0f0'
    ctx.lineWidth = 1
    ctx.stroke()
  }, [currentIndex, losses])

  return (
    <div id="loss-chart">
      <canvas id="loss-canvas" ref={canvasRef} />
    </div>
  )
}

export default LossChart
