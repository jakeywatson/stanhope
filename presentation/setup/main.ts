import { defineAppSetup } from '@slidev/types'
// @ts-expect-error deep import into Slidev internals
import { showGotoDialog, showOverview } from '@slidev/client/state/index.ts'

// Force the overview/menu overlay to always close on Escape or `o`,
// regardless of focus or drawing-mode state. The stock Slidev shortcut
// guards both keys with `not(drawingEnabled)`, which sometimes leaves
// the overlay stuck open; a capture-phase document listener bypasses it.
export default defineAppSetup(() => {
  if (typeof document === 'undefined') return

  const isEditable = (el: EventTarget | null): boolean => {
    if (!(el instanceof HTMLElement)) return false
    const tag = el.tagName
    return (
      tag === 'INPUT' ||
      tag === 'TEXTAREA' ||
      tag === 'SELECT' ||
      el.isContentEditable
    )
  }

  document.addEventListener(
    'keydown',
    (e) => {
      if (isEditable(e.target)) return
      if (e.key === 'Escape') {
        if (showOverview.value || showGotoDialog.value) {
          showOverview.value = false
          showGotoDialog.value = false
          e.preventDefault()
          e.stopPropagation()
        }
        return
      }
      if (
        (e.key === 'o' || e.key === 'O' || e.key === '`') &&
        !e.metaKey &&
        !e.ctrlKey &&
        !e.altKey
      ) {
        showOverview.value = !showOverview.value
        e.preventDefault()
        e.stopPropagation()
      }
    },
    true,
  )
})
