require 'math'
require 'image'
require 'string'
local tablex = require('pl.tablex')
local path = require('pl.path')
--[[
Crystalsort organizes elements in a 2D table, maximizing some score that depends
on the relative positions of each element.

Assuming that dimension 1 is rows and dimension 2 is columns:

Crystal sort will only move an element within its starting column. Elements will
never be moved to a new column. However, the score can take into account
elements in other columns.

In the documentation/code below, each column has a set of elements that have
not yet been assigned a new position. "Layer" refers to the (ordered) collection
of elements that have been assigned a position; "set" refers to the (unordered)
collection of elements that have not been assigned a position.

This is the algorithm:

1. Assign an arbitrary element to layer 1 from its set.
2. Repeat the following steps until every element is inserted:
  a. Select a layer to insert into.
    i. The layer is chosen randomly. Each layer has a chance of selection equal
       to the size of its set, divided by the sum of all layers' set sizes.
  b. Choose the best element to insert.
    i. For each element in that layer's set, test each possible position for 
       insertion. Elements can be inserted only adjacent to elements that have 
       already been inserted. "Adjacency" is considered across both rows and
       columns.
    j. "Best" is defined as the insertion that maximizes the overall score. The
       overall score is the sum of the scores computed for each element in each
       layer.
       
Note that this was developed with neural nets in mind so there's some of that
perspective floating around, e.g. the visualization thinks "depth" as the
number of columns and "width" as the number of rows, analogous to layers in
a neural net and the number of neurons per layer.
--]]

local Crystal = torch.class('Crystal')
function Crystal:__init(width, depth, elements, relations)
  -- Width is the number of rows, depth is the number of columns. elements is
  -- a 2D table of the crystal's elements and relations is a 4D table containing
  -- each element's relationship to each other element. If elements is MxN, then
  -- relations is MxNxMxN.
  self.structure, self.placeholder = {{1}}, {{1}}
  -- self.structure contains a table for each row, listing the elements in that
  -- row for each column. Initially, an arbitrary element of layer 1 is 
  -- assigned position 1. Empty positions have value nil. self.placeholder is
  -- used for testing each insertion possibility

  self.sets = {} -- Table of all unassigned elements, organized by column
  for i=1,depth do
    self.sets[i] = {}
    for j=1,width do
      self.sets[i][j] = j
    end
  end
  table.remove(self.sets[1], 1) -- Element 1 of layer 1 is already inserted

  self.width = width
  self.depth = depth
  self.relations = relations
  self.mean_relation = torch.mean(relations)
  self.elements = elements
  self.visual = true -- Hardcoded because lazy demo
  self.step = 0
end

function Crystal:build()
  -- The main method for actually building the crystal.
  local remaining_elements = 0 -- total number of uninserted elements
  for i=1,#self.sets do
    remaining_elements = remaining_elements + #self.sets[i]
  end


  -- available_elements is the number of elements in columns that can have an
  -- element inserted into them, i.e. columns that have at least 1 element or
  -- are adjacent to at a column with at least 1 element. Initially this is
  -- columns 1 or 2.
  local available_elements = #self.sets[1] + #self.sets[2]

  -- Build loop proper
  while remaining_elements > 0 do
    -- Select a column at random to insert into. Divides a range up into 1 
    -- region for each column that can be inserted into, with each region
    -- proportional to the number of elements left in the column's set. Then
    -- pick a random point in the range and see which region it's in.

    local point = math.random(available_elements)
    local layer
    for i=1,#self.sets do
      point = point - #self.sets[i]
      if point <= 1 and #self.sets[i] > 0 then
        layer = i
        break
      end
    end

    self:insert(layer)
    self.step = self.step+1
    if self.visual then
      self:visualize()
    end

    -- Recalculate available_elements. Could probably be improved considerably.
    available_elements = 0
    for i=1,#self.sets do
      available_elements = available_elements + #self.sets[i]
      if #self.sets[i] == self.width then
        break
      end
    end
    remaining_elements = remaining_elements - 1
  end

  -- Display loop
  for i=1,self.depth do
    for j=1,self.width do
      if self.structure[j] and self.structure[j][i] then
        io.write(tostring(self.structure[j][i]) .. ",\t")
      else
        io.write(tostring("nil,\t"))
      end
    end
    io.write("\n")
  end
  io.write("\n")
end

function Crystal:insert(column) 
  -- Determines every possible location for insertion of a new element into
  -- the given column, then chooses the insertion that maximizes the score.
  -- Assumes that elements can only be inserted adjacent to other elements.
  
  local best_score = -math.huge -- Neg. infinity ensures insertion every time
  local best_result = nil
  local best_address = nil
  local best_position = nil
  local inserted = false


  -- potential_insertion_points is a list of all unoccupied spaces in the
  -- crystal. If the crystal hasn't reached maximum with, 0 is added to shift
  -- the crystal downward. Likewise, it's possible to insert 1 position past
  -- the last element in the layer if the crystal isn't max-width.

  local potential_insertion_points = {}

  if #self.structure < self.width then
    if self.structure[1][column] then
      table.insert(potential_insertion_points, 0)
    end
    if self.structure[#self.structure][column] then
      table.insert(potential_insertion_points, #self.structure+1)
    end  
  end

  for row=1,#self.structure do
    if self.structure[row][column] == nil then
      if self.structure[row][column-1] or 
         self.structure[row][column+1] or
         (row > 1 and self.structure[row-1][column]) or
         (row < #self.structure and self.structure[row+1][column]) then
        table.insert(potential_insertion_points, row)
      end
    end
  end

  for _,row in ipairs(potential_insertion_points) do
    for address,candidate in ipairs(self.sets[column]) do
      self.placeholder = tablex.deepcopy(self.structure) -- paranoid deepcopy
      if row == 0 then
        table.insert(self.placeholder, 1, {})
        self.placeholder[1][column] = candidate
        inserted = true
      elseif row == #self.structure+1 then
        table.insert(self.placeholder, {})
        self.placeholder[#self.placeholder][column] = candidate
      else
        self.placeholder[row][column] = candidate
        inserted = true
      end
      
      -- If the insertion row is "0" (before the first row), a row is inserted.
      -- Therefore after the insertion, the new element will be in row 1.
      -- Otherwise coordinates are unchanged.
      
      local test_score = self:score(column, math.max(1, row))
      if test_score > best_score then
        best_score = test_score
        best_result = tablex.deepcopy(self.placeholder)
        best_address = address
        best_position = row
      end
    end
  end

  print("Inserted", self.sets[column][best_address], "into layer", column, "at", best_position)

  self.structure = best_result
  table.remove(self.sets[column], best_address)
end

function Crystal:score(column, row)
  -- Computes the score for a given insertion by summing the relationship
  -- scores between the newly-inserted element and every element within
  -- influence_size (4 here) steps of that element (Manhattan distance).
  -- Each relationship score is scaled by a weight that decreases with
  -- (Manhattan) distance between the newly-inserted element and the element
  -- whose score contribution is being calculated.
  
  local score = 0
  local power = 1.5
  local influence_size = 4
  local number_of_rows = #self.placeholder
  local max_col_to_check = math.min(column+influence_size, self.depth)
  local min_col_to_check = math.max(1, column-influence_size)
  local max_row_to_check = math.min(row+influence_size, number_of_rows)
  local min_row_to_check = math.min(1, row-influence_size)
  
  -- We need two offsets into the relations matrix - offset_for_inserted is
  -- the offset for the newly-inserted element, while offset is computed for
  -- each element being compared to the newly-inserted element. A given
  -- element's location on each axis of the relations table is offset + the
  -- element's original row number in the input.
  
  local offset_for_inserted = self.width * (column - 1)
  local inserted_location = offset_for_inserted + self.placeholder[row][column]
  for curr_col=min_col_to_check,max_col_to_check do
    for curr_row=min_row_to_check,max_row_to_check do
      local offset = self.width * (curr_col-1)
      local relation_score
      if self.placeholder[curr_row] and self.placeholder[curr_row][curr_col] then
        local contributor = offset + self.placeholder[curr_row][curr_col]
        relation_score = self.relations[inserted_location][contributor]
      else
        relation_score = self.mean_relation
      end
      local weight = math.pow(math.abs(curr_col-column)+math.abs(curr_row-row)+1,
                              power)
      score = score + relation_score/weight
    end
  end
  return score
end

function Crystal:visualize()
  -- Prepares a PNG representation of the current state of the algorithm, 
  -- showing the un-sorted elements of the crystal on the left and the sorted
  -- elements on right. Note that this is heavily domain-dependent and the
  -- visual representation you would actually want will vary - this should
  -- probably therefore be a stub method that calls a visualization function
  -- passed along with the object creation. But this is lame demo code so it's
  -- not.
  
  -- The loop below copies the pixel values out of the [row][col][channel] 3D
  -- table that was used in the algorithm above to the [channel][row][col] 
  -- tensor expected by the PNG-writing library. This first loop operates on
  -- the original, unsorted inputs.
  local output = torch.Tensor(3, self.width, self.depth*2):zero()
  for i=1,self.width do
    for layer=1,self.depth do
      for channel=1,3 do
        output[channel][i][layer] = self.elements[layer][i][channel]
      end
    end
  end
  
  -- The row below does the same thing with the sorted output. Note that 
  -- self.structure[row][layer] is used to get the ORIGINAL row of the element 
  -- that appears at the row in the output. It also blacks out the appropriate
  -- pixel in the unsorted input. I'm almost positive these loops can be 
  -- combined but too lazy to reason through it.  
  
  for row=1,self.width do
    for layer=1,self.depth do
      if self.structure[row] and self.structure[row][layer] then
        for channel=1,3 do
          local val = self.elements[layer][self.structure[row][layer]][channel]
          output[channel][row][layer+self.depth] = val
          output[channel][self.structure[row][layer]][layer] = 0
        end
      end
    end
  end

  local prefix = ""
  for digit=1,(10-string.len(tostring(self.step))) do
    prefix = prefix .. tostring(0)
  end
  local pth = path.expanduser('~/crystal_pics/img'..prefix..tostring(self.step)..'.png')
  image.save(pth, output)
end

-- Finally the actual demo - this gins up a rectangle of random pixels, then
-- uses crystalsort to sort them.

local rectwid = 32
local rectlen = 64
local randsquare = {}

-- The actual algorithm expects a table for its elements so this generates the
-- random pixels with torch then copies them into a table. Pretty dumb.
local rand_matrix = torch.rand(rectwid, rectlen, 3)
for i=1, rectwid do
  randsquare[i] = {}
  for j=1,rectlen do
    randsquare[i][j] = rand_matrix[i][j]:clone()
  end
end

-- This next chunk reshapes the matrix above to a 1D "vector" of pixels.
-- Each pixel has 3 color values but it's easier to think of it this way.
-- Once we have the 1D vector, it's cloned to make a square matrix where each
-- row is a copy of the vector. Then transpose the square matrix and subtract
-- the original vector from each of the new rows - a very clunky way of getting
-- a matrix of the difference between each pixels. Finally, define the relations
-- matrix as the 2-norm (euclidean distance) of each pixel's values.
local rand_vector = torch.reshape(rand_matrix, rectwid*rectlen, 3)
local diffs_matrix = torch.Tensor(rand_vector:size()[1], rand_vector:size()[1], rand_vector:size()[2]):zero()
for i=1,rand_vector:size()[1] do
  diffs_matrix[i] = rand_vector:clone()
end
diffs_matrix = diffs_matrix:transpose(1, 2)
for i=1,rand_vector:size()[1] do
  diffs_matrix[i] = diffs_matrix[i]-rand_vector
end
local relations = torch.norm(diffs_matrix, 2, 3):squeeze()

-- Then to finish the whole thing up, invert the signs so that more-different
-- pixels have lower scores.
relations:mul(-1)

local randcryst = Crystal.new(rectlen, rectwid, randsquare, relations)
randcryst:build()