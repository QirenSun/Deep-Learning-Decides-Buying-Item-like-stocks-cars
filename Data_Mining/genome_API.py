import visual_genome.local as vg

r_d,images=vg.get_all_region_descriptions(data_dir= "D:/visual/")

relation=vg.get_all_relationships(data_dir= "D:/visual/")


test=vg.relation_test(data_dir= "D:/visual/")


