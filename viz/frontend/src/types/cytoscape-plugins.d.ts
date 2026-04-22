/**
 * Ambient declarations for cytoscape plugins that don't ship @types.
 *
 * `react-cytoscapejs` exports a default component that takes the same props
 * as cytoscape.Core initialization plus `elements` / `stylesheet` / `cy`.
 * `cytoscape-fcose` is a cytoscape extension registered via `cytoscape.use`.
 */

declare module 'cytoscape-fcose' {
  import type { Ext } from 'cytoscape'
  const fcose: Ext
  export default fcose
}

declare module 'cytoscape-elk' {
  import type { Ext } from 'cytoscape'
  const elk: Ext
  export default elk
}

declare module 'react-cytoscapejs' {
  import type { ComponentType, CSSProperties } from 'react'
  import type { Core, ElementDefinition, StylesheetStyle } from 'cytoscape'

  export interface CytoscapeComponentProps {
    elements: ElementDefinition[]
    stylesheet?: StylesheetStyle[]
    layout?: cytoscape.LayoutOptions
    cy?: (cy: Core) => void
    style?: CSSProperties
    className?: string
    wheelSensitivity?: number
    panningEnabled?: boolean
    userPanningEnabled?: boolean
    zoomingEnabled?: boolean
    userZoomingEnabled?: boolean
    boxSelectionEnabled?: boolean
  }

  const CytoscapeComponent: ComponentType<CytoscapeComponentProps>
  export default CytoscapeComponent
}
