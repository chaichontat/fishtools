export interface IjkManifest {
  version: number;
  source: string;
  coord_space: "ijk";
  position_order: string;
  n_points: number;
  files: {
    positions_f32: string;
    r_um_f32: string;
    t_lookup_f32: string;
    axis_u8: string;
    leiden_u16?: string;
    leiden_categories_json?: string;
  };
  stats: {
    r_um_min: number;
    r_um_max: number;
    t_lookup_min: number;
    t_lookup_max: number;
    bbox_xyz_min: [number, number, number];
    bbox_xyz_max: [number, number, number];
  };
  axis_counts: {
    coronal: number;
    sagittal: number;
  };
  leiden?: {
    n_categories: number;
    counts?: Record<string, number>;
  };
  export: Record<string, unknown>;
}

export interface LoadedIjkAssets {
  manifest: IjkManifest;
  positions: Float32Array;
  rUm: Float32Array;
  tLookup: Float32Array;
  axis: Uint8Array;
  leiden?: Uint16Array;
  leidenCategories?: string[];
}
