function [guidewire_positions, oct_volume_guidewire_removed] = remove_guidewire_block(oct_volume)

    [guidewire_positions, oct_volume_guidewire_removed, ~, ~] = removeGuideWire(oct_volume);    

end