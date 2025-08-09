'use server';

import { summarizeProject, type SummarizeProjectInput } from '@/ai/flows/project-summarizer';
import { z } from 'zod';

const SummarizeActionInput = z.object({
  projectDescription: z.string().min(20, 'Please provide a more detailed project description (at least 20 characters).'),
  audienceType: z.enum(['technical', 'non-technical']),
});

type SummarizeActionState = {
  success: boolean;
  message?: string;
  summary?: string;
};

export async function summarizeProjectAction(
  prevState: SummarizeActionState,
  formData: FormData
): Promise<SummarizeActionState> {
  
  const validatedFields = SummarizeActionInput.safeParse({
    projectDescription: formData.get('projectDescription'),
    audienceType: formData.get('audienceType'),
  });

  if (!validatedFields.success) {
    return {
      success: false,
      message: validatedFields.error.errors.map((e) => e.message).join(', '),
    };
  }

  try {
    const result = await summarizeProject(validatedFields.data);
    return {
      success: true,
      summary: result.summary,
    };
  } catch (error) {
    console.error('Error summarizing project:', error);
    return {
      success: false,
      message: 'An unexpected error occurred. Please try again.',
    };
  }
}
