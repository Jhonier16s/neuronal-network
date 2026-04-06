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
    ctx.fillStyle = 'rgba(8, 18, 26, 0.8)'
    ctx.fillRect(0, 0, width, height)

    if (losses.length < 2) {
      return
    }

    let maxLoss = Math.max(...losses)

    if (maxLoss < 0.001) {
      maxLoss = 0.001
    }

    ctx.strokeStyle = 'rgba(58, 92, 116, 0.45)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, height / 2)
    ctx.lineTo(width, height / 2)
    ctx.stroke()

    const gradient = ctx.createLinearGradient(0, 0, 0, height)
    gradient.addColorStop(0, 'rgba(142, 234, 255, 0.28)')
    gradient.addColorStop(1, 'rgba(142, 234, 255, 0)')

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

    ctx.lineTo(width, height - 2)
    ctx.lineTo(0, height - 2)
    ctx.closePath()
    ctx.fillStyle = gradient
    ctx.fill()

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

    ctx.strokeStyle = '#78deff'
    ctx.lineWidth = 2
    ctx.shadowColor = '#78deff'
    ctx.shadowBlur = 10
    ctx.stroke()
    ctx.shadowBlur = 0

    const markerX = (currentIndex / Math.max(losses.length - 1, 1)) * width
    const markerY =
      height - (losses[Math.min(currentIndex, losses.length - 1)] / maxLoss) * (height - 4) - 2

    ctx.beginPath()
    ctx.moveTo(markerX, 0)
    ctx.lineTo(markerX, height)
    ctx.strokeStyle = 'rgba(255, 211, 138, 0.8)'
    ctx.lineWidth = 1
    ctx.stroke()

    ctx.beginPath()
    ctx.arc(markerX, markerY, 3.5, 0, Math.PI * 2)
    ctx.fillStyle = '#ffd38a'
    ctx.fill()
  }, [currentIndex, losses])

  return (
    <div id="loss-chart">
      <canvas id="loss-canvas" ref={canvasRef} />
    </div>
  )
}

export default LossChart
