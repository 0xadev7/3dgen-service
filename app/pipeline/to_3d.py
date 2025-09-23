def single_view_mesh(triposr_model, rgba_img, out_obj_path, opt):
    mesh = triposr_model.infer_mesh(rgba_img)   # returns trimesh.Trimesh
    # optional simplification to keep it fast
    mesh = triposr_model.simplify(mesh,
                                  target_faces=opt.to3d.simplify_to)
    mesh.export(out_obj_path)
