// @ts-expect-error react-cytoscapejs has no type declarations
import CytoscapeComponent from 'react-cytoscapejs'
import cytoscape from 'cytoscape'
// @ts-expect-error cytoscape-fcose has no type declarations
import fcose from 'cytoscape-fcose'
import type { CytoscapeElement } from '@/lib/transforms/cytoscape'
import type { LayoutName } from '@/lib/constants'

// Register fCoSE layout extension once at module level
cytoscape.use(fcose)

interface GraphCanvasProps {
  elements: CytoscapeElement[]
  layout?: LayoutName
  layoutOptions?: Record<string, unknown>
  onNodeSelect?: (nodeId: string) => void
  onEdgeSelect?: (edgeId: string) => void
  className?: string
  cyRef?: React.MutableRefObject<cytoscape.Core | null>
}

const fullStylesheet: cytoscape.StylesheetStyle[] = [
  {
    selector: 'node',
    style: {
      'background-color': 'data(color)' as any,
      label: 'data(label)',
      width: 'data(size)' as any,
      height: 'data(size)' as any,
      'font-size': '8px',
      'text-valign': 'bottom',
      'text-margin-y': 3,
      color: '#e5e7eb',
      'text-outline-width': 2,
      'text-outline-color': '#1f2937',
    },
  },
  {
    selector: 'node[?highlighted]',
    style: {
      'border-width': 2,
      'border-color': '#ff6600',
    },
  },
  {
    selector: 'edge',
    style: {
      width: 2,
      'line-color': '#999',
      'curve-style': 'bezier',
      'target-arrow-shape': 'triangle',
      'target-arrow-color': '#999',
      'arrow-scale': 0.6,
      label: 'data(rel_type)' as any,
      'font-size': '7px',
      'text-rotation': 'autorotate' as any,
      color: '#777',
      'text-margin-y': -8,
    },
  },
  {
    selector: 'edge.highlighted',
    style: {
      'line-color': '#ff6600',
      'target-arrow-color': '#ff6600',
      width: 4,
    },
  },
  {
    selector: ':parent',
    style: {
      'background-opacity': 0.05,
      'border-width': 1,
      'border-color': '#aaa',
      'border-opacity': 0.4,
      label: 'data(label)',
      'font-size': '10px',
      'text-valign': 'top',
      'text-margin-y': -4,
      color: '#888',
    } as any,
  },
]

const miniStylesheet: cytoscape.StylesheetStyle[] = [
  {
    selector: 'node',
    style: {
      'background-color': 'data(color)' as any,
      label: 'data(label)',
      width: 'data(size)' as any,
      height: 'data(size)' as any,
      'font-size': '6px',
      'text-valign': 'bottom',
      'text-margin-y': 3,
      color: '#888',
    },
  },
  {
    selector: 'edge',
    style: {
      width: 1,
      'line-color': '#ddd',
      'curve-style': 'bezier',
      'target-arrow-shape': 'triangle',
      'target-arrow-color': '#ddd',
      'arrow-scale': 0.4,
    },
  },
]

export function GraphCanvas({
  elements,
  layout = 'cose',
  layoutOptions,
  onNodeSelect,
  onEdgeSelect,
  className = 'h-full',
  cyRef,
}: GraphCanvasProps) {
  const interactive = !!onNodeSelect || !!onEdgeSelect
  const stylesheet = interactive ? fullStylesheet : miniStylesheet

  if (elements.length === 0) {
    return (
      <div className={`${className} flex items-center justify-center text-muted-foreground text-xs`}>No graph data</div>
    )
  }

  return (
    <div className={`${className} relative bg-muted/30 rounded-lg overflow-hidden`}>
      <CytoscapeComponent
        elements={elements as any}
        stylesheet={stylesheet}
        layout={{ name: layout, animate: false, ...layoutOptions } as any}
        style={{ width: '100%', height: '100%' }}
        cy={(cy: cytoscape.Core) => {
          if (cyRef) cyRef.current = cy
          cy.on('layoutstop', () => cy.fit(undefined, 20))
          if (onNodeSelect) {
            cy.on('tap', 'node', (evt: cytoscape.EventObject) => {
              onNodeSelect(evt.target.id())
            })
          }
          if (onEdgeSelect) {
            cy.on('tap', 'edge', (evt: cytoscape.EventObject) => {
              onEdgeSelect(evt.target.id())
            })
          }
        }}
      />
    </div>
  )
}
